import nltk

from model.models import UserModelSession, Choice, UserModelRun, Protocol
from model.classifiers import get_emotion, get_sentence_score
import pandas as pd
import numpy as np
import random
from collections import deque
import re
import datetime
import time

nltk.download("wordnet")
from nltk.corpus import wordnet  # noqa


class ModelDecisionMaker:
    def __init__(self):
        self.mydata = pd.read_csv(
            '/Users/PYT/Desktop/project/code/2/SATbot2.0-main/model/mydata.csv', encoding='ISO-8859-1')
        # Titles from workshops (Title 7 adapted to give more information)
        self.PROTOCOL_TITLES = [
            "0: None",
            "1: Recalling Significant Early Memories",
            "2: Becoming Intimate with our Child",
            "3: Singing a Song of Affection",
            "4: Expressing Love and Care for the Child",
            "5: Pledging to Care and Support Our Child",
            "6: Restoring Our Emotional World after Our Pledge",
            "7: 7a Maintaining a Loving Relationship with the Child",
            "8: 7b Creating Zest for Life",
            "9: Overcoming Current Negative Emotions",
            "10: Overcoming Past Pain",
            "11: Muscle Relaxation and Playful Face",
            "12: Laughing on Our Own",
            "13: xx",  # noqa
            "14: Creating Your Own Brand of Laughter",
            "15: Learning to Change Our Perspective",
            "16: Learning to be Playful about Our Past Pains",
            "17: xx",  # noqa
            "18: xx",  # noqa
            "19: xx",  # noqa
            "20: Practicing Affirmations",
            "21: restrain your desires and sensual pleasure",
            "22: forgo an instance of your bad habits every day",
            "23: strengthen your muscle to strengthen your willpower",
            "24: consume less and lead a simple life",
            "25: read affirmations",
            # "21: Restrain Your Desires and Sensual Pleasure",
            # "22: Forgo an Instance of Your Bad Habits Every Day",
            # "23: Strengthen Your Muscle to Strengthen Your Willpower",
            # "24: Consume Less and Lead a Simple Life",
            # "25: Read Affirmations",
        ]

        self.TITLE_TO_PROTOCOL = {
            self.PROTOCOL_TITLES[i]: i for i in range(len(self.PROTOCOL_TITLES))
        }

        self.recent_protocols = deque(maxlen=26)
        self.reordered_protocol_questions = {}
        self.protocols_to_suggest = []
        self.u100 = "Go ahead with the protocol in your own time."

        # Goes from user id to actual value
        self.current_run_ids = {}
        self.current_protocol_ids = {}

        self.current_protocols = {}

        self.positive_protocols = [1, 2, 3, 4, 5, 6, 7, 20, 9, 12, 14]
        self.specific_ex = [i for i in range(21, 26)]

        self.INTERNAL_PERSECUTOR_PROTOCOLS = [
            self.PROTOCOL_TITLES[15],
            self.PROTOCOL_TITLES[16],
            self.PROTOCOL_TITLES[8],
            self.PROTOCOL_TITLES[19],
        ]

        # Keys: user ids, values: dictionaries describing each choice (in list)
        # and current choice
        self.user_choices = {}

        # Keys: user ids, values: scores for each question
        #self.user_scores = {}

        # Keys: user ids, values: current suggested protocols
        self.suggestions = {}

        # Tracks current emotion of each user after they classify it
        self.user_emotions = {}

        self.guess_emotion_predictions = {}
        # Structure of dictionary: {question: {
        #                           model_prompt: str or list[str],
        #                           choices: {maps user response to next protocol},
        #                           protocols: {maps user response to protocols to suggest},
        #                           }, ...
        #                           }
        # This could be adapted to be part of a JSON file (would need to address
        # mapping callable functions over for parsing).

        self.users_names = {}
        self.remaining_choices = {}

        self.recent_questions = {}

        self.chosen_personas = {}
        self.datasets = {}
        self.suggestion_SAT_P = [1, 2, 3, 4, 5, 6, 7, 20, 9, 12, 14]
        self.suggestion_SPE = [20, 21, 22, 23, 24]
        self.suggestion_nr = [20, 21, 22, 23, 24]
        self.suggestion_nd = [20, 21, 22, 23, 24]
        self.utterance = ["Terrific, I want follow up exercises",
                          "Wow nice, I want follow up exercises",
                          "Nice, I want follow up exercises",
                          "Good, I want follow up exercises",
                          "Cool, I want follow up exercises",
                          "Wonderful, I want follow up exercises",
                          "Neat, I want follow up exercises",
                          "Excellent, I want follow up exercises",
                          "Fabulous, I want follow up exercises,"
                          "Wonderful, I want follow up exercises",
                          "Fabulous, I want follow up exercises",
                          "Terrific, I want follow up exercises",
                          "Awesome, I want follow up exercises",
                          "I love this, I want follow up exercises",
                          "Love it, I want follow up exercises",
                          "Brilliant, I want follow up exercises",
                          "Cool, I want follow up exercises",
                          "I love this, I want follow up exercises",
                          "Love it, I want follow up exercises"
                          ]

        self.QUESTIONS = {

            ############################ Begin Beginning  #########################################
            "ask_name": {
                "model_prompt": "Please enter your first name:",
                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.save_name(user_id)
                },
                "protocols": {"open_text": []},
            },

            "greeting": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_intro(user_id),
                "choices": {
                    "Good, let's start!": "ask_noble_goal",
                },
                "protocols": {"Good, let's start!": []},
            },

            "qa": {
                "model_prompt": "Here is the link: https://testyourself.psychtests.com/testid/3191, when you finish, please press Continue",
                "choices": {
                    "Continue": "ask_noble_goal",
                },
                "protocols": {"Continue": []},
            },

            "ask_noble_goal": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ask_noblegoal(user_id, app, db_session),
                "choices": {
                    "Yes, I currently have a noble goal to pursue.": "opening_prompt",
                    "At the moment, I do not have one.": "set_noble_goal",
                },
                "protocols": {"Yes, I currently have a noble goal to pursue.": [], "At the moment, I do not have one.": [], },
            },

            "set_noble_goal": {
                "model_prompt": "Now's the time to set one! When finish setting your noble goal, you could press the button to continue the conversation.",
                "choices": {
                    "continue": "opening_prompt",
                },
                "protocols": {"continue": []},
            },
            ############################ End Beginning  #########################################
            ############################ Begin Conversation  #########################################
            "opening_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_opening_prompt(user_id),
                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_opening(user_id, app, db_session)
                },
                "protocols": {"open_text": []},
            },

            "guess_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_guess_emotion(
                    user_id, app, db_session
                ),
                "choices": {
                    "yes": {
                        "Sad": "after_classification_negative",
                        "Angry": "after_classification_negative",
                        "Anxious/Scared": "after_classification_negative",
                        "Happy/Content": "after_classification_positive",
                    },
                    "no": "check_emotion",
                },
                "protocols": {
                    "yes": [],
                    "no": []
                },
            },

            "check_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_check_emotion(user_id, app, db_session),
                "choices": {
                    "Sad": lambda user_id, db_session, curr_session, app: self.get_sad_emotion(user_id),
                    "Angry": lambda user_id, db_session, curr_session, app: self.get_angry_emotion(user_id),
                    "Anxious/Scared": lambda user_id, db_session, curr_session, app: self.get_anxious_emotion(user_id),
                    "Happy/Content": lambda user_id, db_session, curr_session, app: self.get_happy_emotion(user_id),
                },
                "protocols": {
                    "Sad": [],
                    "Angry": [],
                    "Anxious/Scared": [],
                    "Happy/Content": []
                },
            },

            ######################### Negative #############################
            "after_classification_negative": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_specific_event(user_id, app, db_session),

                "choices": {
                    "Yes, something happened": "sat_or_25",
                    "No, it's just a general feeling": "add_q",
                },
                "protocols": {
                    "Yes, something happened": [],
                    "No, it's just a general feeling": []
                },
            },

            "sat_or_25": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_sat_or_aff(user_id, app, db_session),

                "choices": {
                    "SAT Exercises": "event_is_recent",
                    "Read Affirmations": "e20",
                },
                "protocols": {
                    "SAT Exercises": [],
                    "Read Affirmations": []
                },
            },

            "add_q": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_add_q(user_id),

                "choices": {
                    "Yes, you could": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "I prefer not": "e15_reluctant",
                },
                "protocols": {
                    "Yes, you could": [],
                    "I prefer not": []
                },
            },



            "event_is_recent": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_event_is_recent(user_id, app, db_session),

                "choices": {
                    "It was recent": "e9_nr",
                    "It was distant": "e10_nd",
                },
                "protocols": {
                    "It was recent": [],
                    "It was distant": []
                },
            },

            "revisiting_recent_events": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_revisit_recent(user_id, app, db_session),

                "choices": {
                    "yes": "more_questions",
                    "no": "more_questions",
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[8], self.PROTOCOL_TITLES[11]],
                    "no": [self.PROTOCOL_TITLES[9]],
                },
            },

            "revisiting_distant_events": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_revisit_distant(user_id, app, db_session),

                "choices": {
                    "yes": "more_questions",
                    "no": "more_questions",
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[15], self.PROTOCOL_TITLES[16]],
                    "no": [self.PROTOCOL_TITLES[10]]
                },
            },

            "more_questions": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_more_questions(user_id, app, db_session),

                "choices": {
                    "Okay": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                    "I'd rather not": "project_emotion",
                },
                "protocols": {
                    "Okay": [],
                    "I'd rather not": [self.PROTOCOL_TITLES[15]],
                },
            },



            ################# POSITIVE EMOTION (HAPPINESS/CONTENT) #################

            "after_classification_positive": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_happy(user_id, app, db_session),
                "choices": {
                    "SAT Exercise": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "Specific Exercises Dedicated to Willpower": lambda user_id, db_session, curr_session, app: self.get_spe_suggestions(user_id),
                    "No, thank you": "ending_prompt",
                },
                "protocols": {
                    "SAT Exercise": [self.PROTOCOL_TITLES[9], self.PROTOCOL_TITLES[12], self.PROTOCOL_TITLES[14]],
                    "Specific Exercises Dedicated to Willpower": [],
                    "No, thank you": []
                },
            },
            "suggestions_P_SAT": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggestions(user_id, app, db_session),

                "choices": {
                    # self.current_protocol_ids[user_id]
                    self.PROTOCOL_TITLES[9]: "try_protocol",
                    self.PROTOCOL_TITLES[12]: "try_protocol",
                    self.PROTOCOL_TITLES[14]: "try_protocol",
                    # self.PROTOCOL_TITLES[k]: "trying_protocol"
                    # for k in self.positive_protocols
                },
                "protocols": {
                    self.PROTOCOL_TITLES[9]: [],
                    self.PROTOCOL_TITLES[12]: [],
                    self.PROTOCOL_TITLES[14]: [],
                    # self.PROTOCOL_TITLES[k]: "trying_protocol"
                    # for k in self.positive_protocols
                },
            },
            "no_more_choice": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.end_of_suggestions(user_id),
                "choices": {
                    "End Session": "ending_prompt",
                    "Try Specific Willpower Exercises": lambda user_id, db_session, curr_session, app: self.get_spe_suggestions(user_id),

                },
                "protocols": {
                    "End Session": [],
                    "Try Specific Willpower Exercises": []
                },
            },
            "no_more_choice_spe": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.end_of_suggestions_spe(user_id),
                "choices": {
                    "End Session": "ending_prompt",
                    "I would like to explore SAT Exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                },
                "protocols": {
                    "End Session": [],
                    "I would like to explore SAT Exercises": []
                },
            },
            "no_more_choice_nr": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.end_of_suggestions_spe(user_id),
                "choices": {
                    "End Session": "ending_prompt",
                    "I would like to explore Specific Willpower Exercises": lambda user_id, db_session, curr_session, app: self.get_spe_suggestions(user_id),
                },
                "protocols": {
                    "End Session": [],
                    "I would like to explore Specific Willpower Exercises": []
                },
            },
            "no_more_choice_nd": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.end_of_suggestions_spe(user_id),
                "choices": {
                    "End Session": "ending_prompt",
                    "I would like to explore Specific Willpower Exercises": lambda user_id, db_session, curr_session, app: self.get_spe_suggestions(user_id),
                },
                "protocols": {
                    "End Session": [],
                    "I would like to explore Specific Willpower Exercises": []
                },
            },
            "e1": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e1_intro(user_id),
                "choices": {"Continue": "user_found_useful_e1"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e2": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e2_intro(user_id),
                "choices": {"Continue": "user_found_useful_e1"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e3": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e3_intro(user_id),
                "choices": {"Continue": "user_found_useful_e1"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e4": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e4_intro(user_id),
                "choices": {"Continue": "user_found_useful_e1"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e5": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e5_intro(user_id),
                "choices": {"Continue": "user_found_useful_e1"},
                "protocols": {
                    "Continue": [],
                },
            },

            "e6": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e6_intro(user_id),
                "choices": {"Continue": "user_found_useful_e1"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e7": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e7_intro(user_id),
                "choices": {"Continue": "user_found_useful_e1"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e8": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e8_intro(user_id),
                "choices": {"Continue": "user_found_useful_e1"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e9": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e9_intro(user_id),
                "choices": {"Continue": "user_found_useful_e1"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e10": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e10_intro(user_id),
                "choices": {"Continue": "user_found_useful_e1"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e11": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e11_intro(user_id),
                "choices": {"Continue": "user_found_useful_e1"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e12": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e12_intro(user_id),
                "choices": {"Continue": "user_found_useful_e1"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e14": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e14_intro(user_id),
                "choices": {"Continue": "user_found_useful_e1"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e15": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e15_intro(user_id),
                "choices": {"Continue": "user_found_useful_e1"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e16": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e16_intro(user_id),
                "choices": {"Continue": "user_found_useful_e1"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e20": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e20_intro(user_id),
                "choices": {"Continue": "user_found_useful_e1"},
                "protocols": {
                    "Continue": [],
                },
            },


            ###################################################################################################################################
            "e21s": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e21_intro(user_id),
                "choices": {"Continue": "user_found_useful_e21"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e22s": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e22_intro(user_id),
                "choices": {"Continue": "user_found_useful_e21"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e23s": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e23_intro(user_id),
                "choices": {"Continue": "user_found_useful_e21"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e24s": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e24_intro(user_id),
                "choices": {"Continue": "user_found_useful_e21"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e20s": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e20_intros(user_id),
                "choices": {"Continue": "user_found_useful_e21"},
                "protocols": {
                    "Continue": [],
                },
            },


            ##############################################################################################################

            "e9_nr": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e9_intro(user_id),
                "choices": {"Continue": "e9_comfort"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e10_nd": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e10_intro(user_id),
                "choices": {"Continue": "e10_comfort"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e9_comfort": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_revisit_recent(user_id, app, db_session),
                "choices": {"Yes, I find it distressing": "e8_nr",
                            "I'm fine with it.": "e8_nr",
                            },
                "protocols": {
                    "Yes, I find it distressing": [],
                    "I'm fine with it.": [],
                },
            },
            "e10_comfort": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_revisit_recent(user_id, app, db_session),
                "choices": {"Yes, I find it distressing": "e15_nd",
                            "I'm fine with it.": "e15_nd",
                            },
                "protocols": {
                    "Yes, I find it distressing": [],
                    "I'm fine with it.": [],
                },
            },
            "e8_nr": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e8_intro(user_id),
                "choices": {"Continue": "user_found_useful_e31"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e15_nd": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e15_intro(user_id),
                "choices": {"Continue": "user_found_useful_e51"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e11_nr": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e11_intro(user_id),
                "choices": {"Continue": "user_found_useful_e41"},
                "protocols": {
                    "Continue": [],
                },
            },
            "e16_nd": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e16_intro(user_id),
                "choices": {"Continue": "user_found_useful_e61"},
                "protocols": {
                    "Continue": [],
                },
            },
            ##############################################################################################################

            "e15_reluctant": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e15_intro(user_id),
                "choices": {"Continue": "user_found_useful_e15_re"},
                "protocols": {
                    "Continue": [],
                },
            },
            "user_found_useful_e15_re": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_found_useful(user_id, app, db_session),

                "choices": {
                    "I feel better": "new_protocol_better_e15",
                    "I feel worse": "new_protocol_worse_e15",
                    "I feel no change": "new_protocol_same_e15",
                },
                "protocols": {
                    "I feel better": [],
                    "I feel worse": [],
                    "I feel no change": []
                },
            },
            "new_protocol_better_e15": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_better(user_id, app, db_session),

                "choices": {
                    "Neat, I want follow up exercises": "project_e15",
                    "I would like to restart the question": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Neat, I want follow up exercises": [],
                    "I would like to restart the question": [],
                    "I prefer to end this session.": []
                },
            },
            "new_protocol_worse_e15": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_worse(user_id, app, db_session),

                "choices": {
                    "Neat, I want follow up exercises": "project_e15",
                    "I would like to restart the question": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Neat, I want follow up exercises": [],
                    "I would like to restart the question": [],
                    "I prefer to end this session.": []
                },
            },
            "new_protocol_same_e15": {
                "model_prompt": [
                    "I am sorry to hear you have not detected any change in your mood.",
                    "That can sometimes happen but if you agree we could try another protocol and see if that is more helpful to you.",
                    "Would you like me to suggest a different protocol?"
                ],
                "choices": {
                    "Excellent, I want follow up exercises": "project_e15",
                    "Yes (restart questions)": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Excellent, I want follow up exercises": [],
                    "Yes (restart questions)": [],
                    "I prefer to end this session.": []
                },
            },
            "project_e15": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_projectemotion(user_id),

                "choices": {
                    "Continue": "user_found_useful_e1"
                },
                "protocols": {
                    "Continue": []

                },
            },


            ##############################################################################################################

            "try_protocol": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_trying_protocol(user_id, app, db_session),

                "choices": {"continue": "user_found_useful"},
                "protocols": {"continue": []},
            },



            "suggestions_will": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggestions(user_id, app, db_session),
                "choices": {
                    self.PROTOCOL_TITLES[9]: "trying_protocol"
                },
                "protocols": {
                    self.PROTOCOL_TITLES[9]
                },
            },
            "user_found_useful_e1": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_found_useful(user_id, app, db_session),

                "choices": {
                    "I feel better": "new_protocol_better_e1",
                    "I feel worse": "new_protocol_worse_e1",
                    "I feel no change": "new_protocol_same_e1",
                },
                "protocols": {
                    "I feel better": [],
                    "I feel worse": [],
                    "I feel no change": []
                },
            },
            "user_found_useful_e21": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_found_useful(user_id, app, db_session),

                "choices": {
                    "I feel better": "new_protocol_better_e21",
                    "I feel worse": "new_protocol_worse_e21",
                    "I feel no change": "new_protocol_same_e21",
                },
                "protocols": {
                    "I feel better": [],
                    "I feel worse": [],
                    "I feel no change": []
                },
            },
            "user_found_useful_e31": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_found_useful(user_id, app, db_session),

                "choices": {
                    "I feel better": "new_protocol_better_e31",
                    "I feel worse": "new_protocol_worse_e31",
                    "I feel no change": "new_protocol_same_e31",
                },
                "protocols": {
                    "I feel better": [],
                    "I feel worse": [],
                    "I feel no change": []
                },
            },
            "user_found_useful_e51": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_found_useful(user_id, app, db_session),

                "choices": {
                    "I feel better": "new_protocol_better_e51",
                    "I feel worse": "new_protocol_worse_e51",
                    "I feel no change": "new_protocol_same_e51",
                },
                "protocols": {
                    "I feel better": [],
                    "I feel worse": [],
                    "I feel no change": []
                },
            },
            "user_found_useful_e41": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_found_useful(user_id, app, db_session),

                "choices": {
                    "I feel better": "new_protocol_better_e41",
                    "I feel worse": "new_protocol_worse_e41",
                    "I feel no change": "new_protocol_same_e41",
                },
                "protocols": {
                    "I feel better": [],
                    "I feel worse": [],
                    "I feel no change": []
                },
            },
            "user_found_useful_e61": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_found_useful(user_id, app, db_session),

                "choices": {
                    "I feel better": "new_protocol_better_e61",
                    "I feel worse": "new_protocol_worse_e61",
                    "I feel no change": "new_protocol_same_e61",
                },
                "protocols": {
                    "I feel better": [],
                    "I feel worse": [],
                    "I feel no change": []
                },
            },
            ############################# ALL EMOTIONS #############################

            "project_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_project_emotion(user_id, app, db_session),
                "choices": {
                    "continue": "suggestions",
                },
                "protocols": {
                    "continue": [],
                },
            },
            ################################################################################################################################

            "new_protocol_better_e1": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_better(user_id, app, db_session),

                "choices": {
                    "Neat, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "I would like to restart the question": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Neat, I want follow up exercises": [],
                    "I would like to restart the question": [],
                    "I prefer to end this session.": []
                },
            },

            "new_protocol_worse_e1": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_worse(user_id, app, db_session),
                "choices": {
                    "Excellent, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "Yes (restart questions)": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Excellent, I want follow up exercises": [],
                    "Yes (restart questions)": [],
                    "I prefer to end this session.": []
                },
            },

            "new_protocol_same_e1": {
                "model_prompt": [
                    "I am sorry to hear you have not detected any change in your mood.",
                    "That can sometimes happen but if you agree we could try another protocol and see if that is more helpful to you.",
                    "Would you like me to suggest a different protocol?"
                ],
                "choices": {
                    "Excellent, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "Yes (restart questions)": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Excellent, I want follow up exercises": [],
                    "Yes (restart questions)": [],
                    "I prefer to end this session.": []
                },
            },


            ############################################################################################################################
            "new_protocol_better_e21": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_better(user_id, app, db_session),

                "choices": {
                    "Neat, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_spe_suggestions(user_id),
                    "I would like to restart the question": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Neat, I want follow up exercises": [],
                    "I would like to restart the question": [],
                    "I prefer to end this session.": []
                },
            },

            "new_protocol_worse_e21": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_worse(user_id, app, db_session),
                "choices": {
                    "Excellent, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_spe_suggestions(user_id),
                    "Yes (restart questions)": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Excellent, I want follow up exercises": [],
                    "Yes (restart questions)": [],
                    "I prefer to end this session.": []
                },
            },

            "new_protocol_same_e21": {
                "model_prompt": [
                    "I am sorry to hear you have not detected any change in your mood.",
                    "That can sometimes happen but if you agree we could try another protocol and see if that is more helpful to you.",
                    "Would you like me to suggest a different protocol?"
                ],
                "choices": {
                    "Excellent, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_spe_suggestions(user_id),
                    "Yes (restart questions)": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Excellent, I want follow up exercises": [],
                    "Yes (restart questions)": [],
                    "I prefer to end this session.": []
                },
            },






            ################################################################################################################################

            "new_protocol_better_e31": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_better(user_id, app, db_session),

                "choices": {
                    "Neat, I want follow up exercises": "e11_nr",
                    "I would like to restart the question": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Neat, I want follow up exercises": [],
                    "I would like to restart the question": [],
                    "I prefer to end this session.": []
                },
            },

            "new_protocol_worse_e31": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_worse(user_id, app, db_session),
                "choices": {
                    "Excellent, I want follow up exercises": "e11_nr",
                    "Yes (restart questions)": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Excellent, I want follow up exercises": [],
                    "Yes (restart questions)": [],
                    "I prefer to end this session.": []
                },
            },

            "new_protocol_same_e31": {
                "model_prompt": [
                    "I am sorry to hear you have not detected any change in your mood.",
                    "That can sometimes happen but if you agree we could try another protocol and see if that is more helpful to you.",
                    "Would you like me to suggest a different protocol?"
                ],
                "choices": {
                    "Excellent, I want follow up exercises": "e11_nr",
                    "Yes (restart questions)": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Excellent, I want follow up exercises": [],
                    "Yes (restart questions)": [],
                    "I prefer to end this session.": []
                },
            },
            ################################################################################################################################

            "new_protocol_better_e51": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_better(user_id, app, db_session),

                "choices": {
                    "Neat, I want follow up exercises": "e16_nd",
                    "I would like to restart the question": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Neat, I want follow up exercises": [],
                    "I would like to restart the question": [],
                    "I prefer to end this session.": []
                },
            },

            "new_protocol_worse_e51": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_worse(user_id, app, db_session),
                "choices": {
                    "Excellent, I want follow up exercises": "e16_nd",
                    "Yes (restart questions)": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Excellent, I want follow up exercises": [],
                    "Yes (restart questions)": [],
                    "I prefer to end this session.": []
                },
            },

            "new_protocol_same_e51": {
                "model_prompt": [
                    "I am sorry to hear you have not detected any change in your mood.",
                    "That can sometimes happen but if you agree we could try another protocol and see if that is more helpful to you.",
                    "Would you like me to suggest a different protocol?"
                ],
                "choices": {
                    "Excellent, I want follow up exercises": "e16_nd",
                    "Yes (restart questions)": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Excellent, I want follow up exercises": [],
                    "Yes (restart questions)": [],
                    "I prefer to end this session.": []
                },
            },
            ###########################################################################################################################################
            "new_protocol_better_e41": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_better(user_id, app, db_session),

                "choices": {
                    "Neat, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "I would like to restart the question": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Neat, I want follow up exercises": [],
                    "I would like to restart the question": [],
                    "I prefer to end this session.": []
                },
            },

            "new_protocol_worse_e41": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_worse(user_id, app, db_session),
                "choices": {
                    "Excellent, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "Yes (restart questions)": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Excellent, I want follow up exercises": [],
                    "Yes (restart questions)": [],
                    "I prefer to end this session.": []
                },
            },

            "new_protocol_same_e41": {
                "model_prompt": [
                    "I am sorry to hear you have not detected any change in your mood.",
                    "That can sometimes happen but if you agree we could try another protocol and see if that is more helpful to you.",
                    "Would you like me to suggest a different protocol?"
                ],
                "choices": {
                    "Excellent, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "Yes (restart questions)": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Excellent, I want follow up exercises": [],
                    "Yes (restart questions)": [],
                    "I prefer to end this session.": []
                },
            },
            ###########################################################################################################################################
            "new_protocol_better_e61": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_better(user_id, app, db_session),

                "choices": {
                    "Neat, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "I would like to restart the question": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Neat, I want follow up exercises": [],
                    "I would like to restart the question": [],
                    "I prefer to end this session.": []
                },
            },

            "new_protocol_worse_e61": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_worse(user_id, app, db_session),
                "choices": {
                    "Excellent, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "Yes (restart questions)": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Excellent, I want follow up exercises": [],
                    "Yes (restart questions)": [],
                    "I prefer to end this session.": []
                },
            },

            "new_protocol_same_e61": {
                "model_prompt": [
                    "I am sorry to hear you have not detected any change in your mood.",
                    "That can sometimes happen but if you agree we could try another protocol and see if that is more helpful to you.",
                    "Would you like me to suggest a different protocol?"
                ],
                "choices": {
                    "Excellent, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "Yes (restart questions)": "restart_prompt",
                    "I prefer to end this session.": "ending_prompt",
                },
                "protocols": {
                    "Excellent, I want follow up exercises": [],
                    "Yes (restart questions)": [],
                    "I prefer to end this session.": []
                },
            },
            ############################################################################################################################################
            "ending_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ending(user_id, app, db_session),

                "choices": {"any": "opening_prompt"},
                "protocols": {"any": []}
            },

            "restart_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_restart_prompt(user_id),

                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_opening(user_id, app, db_session)
                },
                "protocols": {"open_text": []},
            },
        }
        self.QUESTION_KEYS = list(self.QUESTIONS.keys())

    def initialise_prev_questions(self, user_id):
        self.recent_questions[user_id] = []

    def clear_persona(self, user_id):
        self.chosen_personas[user_id] = ""

    def clear_names(self, user_id):
        self.users_names[user_id] = ""

    def clear_datasets(self, user_id):
        self.datasets[user_id] = pd.DataFrame(columns=['sentences'])

    def initialise_remaining_choices(self, user_id):
        self.remaining_choices[user_id] = []

    def save_name(self, user_id):
        try:
            user_response = self.user_choices[user_id]["choices_made"]["ask_name"]
        except:  # noqa
            user_response = ""
        self.users_names[user_id] = user_response
        # return "choose_persona"
        self.datasets[user_id] = self.mydata
        return "greeting"

    ####################################################################################################

    # from all the lists of protocols collected at each step of the dialogue it puts together some and returns these as suggestions
    def get_new_suggestions(self, user_id):
        if self.suggestion_SAT_P == []:
            return "no_more_choice"
        else:
            selected_choice = np.random.choice(self.suggestion_SAT_P)
            self.suggestion_SAT_P.remove(selected_choice)
            if selected_choice == 1:
                return "e1"
            if selected_choice == 2:
                return "e2"
            if selected_choice == 3:
                return "e3"
            if selected_choice == 4:
                return "e4"
            if selected_choice == 5:
                return "e5"
            if selected_choice == 6:
                return "e6"
            if selected_choice == 7:
                return "e7"
            if selected_choice == 9:
                return "e9"
            if selected_choice == 12:
                return "e12"
            if selected_choice == 14:
                return "e14"
            if selected_choice == 20:
                return "e20"

    def get_spe_suggestions(self, user_id):
        if self.suggestion_SPE == []:
            return "no_more_choice_spe"
        else:
            selected_choice = np.random.choice(self.suggestion_SPE)
            self.suggestion_SPE.remove(selected_choice)
            if selected_choice == 21:
                return "e21s"
            if selected_choice == 20:
                return "e20s"
            if selected_choice == 22:
                return "e22s"
            if selected_choice == 23:
                return "e23s"
            if selected_choice == 24:
                return "e24s"

    def get_nd_suggestions(self, user_id):
        if self.suggestion_SPE == []:
            return "no_more_choice_nd"
        else:
            selected_choice = np.random.choice(self.suggestion_SPE)
            self.suggestion_SPE.remove(selected_choice)
            if selected_choice == 21:
                return "e21s"
            if selected_choice == 20:
                return "e20s"
            if selected_choice == 22:
                return "e22s"
            if selected_choice == 23:
                return "e23s"
            if selected_choice == 24:
                return "e24s"

    def end_of_suggestions(self, user_id):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Amazing! You have tried out all the exercises that I could come up with."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        self.suggestion_SAT_P = [1, 2, 3, 4, 5, 6, 7, 20, 9, 12, 14]
        return [self.split_sentence(question), "You could choose to end session or try specific willpower exercises."]

    def end_of_suggestions_spe(self, user_id):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Amazing! You have tried out all the exercises that I could come up with."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        self.suggestion_SPE = [20, 21, 22, 23, 24]
        return [self.split_sentence(question), "You could choose to end session or try SAT Exercises."]

    def get_suggestions_P_SAT(self, user_id, app):
        suggestions = [9, 12, 14]
        return suggestions

    def get_suggestions_N_SAT_R(self, user_id, app):
        suggestions = [self.PROTOCOL_TITLES[9, 8, 11]]
        for curr_suggestions in list(self.suggestions[user_id]):
            if len(curr_suggestions) > 2:
                i, j = random.choices(range(0, len(curr_suggestions)), k=2)
                # weeds out some gibberish that im not sure why it's there
                if curr_suggestions[i] and curr_suggestions[j] in self.PROTOCOL_TITLES:
                    suggestions.extend(
                        [curr_suggestions[i], curr_suggestions[j]])
            else:
                suggestions.extend(curr_suggestions)
            suggestions = set(suggestions)
            suggestions = list(suggestions)
        # augment the suggestions if less than 4, we add random ones avoiding repetitions
        while len(suggestions) < 4:
            p = random.choice([1, 2, 3, 4, 5, 6, 7, 20])
            if (any(self.PROTOCOL_TITLES[p] not in curr_suggestions for curr_suggestions in list(self.suggestions[user_id]))
                    and self.PROTOCOL_TITLES[p] not in self.recent_protocols and self.PROTOCOL_TITLES[p] not in suggestions):
                suggestions.append(self.PROTOCOL_TITLES[p])
                self.suggestions[user_id].extend([self.PROTOCOL_TITLES[p]])
        return suggestions

    def get_suggestions_N_SAT_D(self, user_id, app):
        suggestions = [self.PROTOCOL_TITLES[10, 15, 16]]
        for curr_suggestions in list(self.suggestions[user_id]):
            if len(curr_suggestions) > 2:
                i, j = random.choices(range(0, len(curr_suggestions)), k=2)
                # weeds out some gibberish that im not sure why it's there
                if curr_suggestions[i] and curr_suggestions[j] in self.PROTOCOL_TITLES:
                    suggestions.extend(
                        [curr_suggestions[i], curr_suggestions[j]])
            else:
                suggestions.extend(curr_suggestions)
            suggestions = set(suggestions)
            suggestions = list(suggestions)
        # augment the suggestions if less than 4, we add random ones avoiding repetitions
        while len(suggestions) < 4:
            p = random.choice([1, 2, 3, 4, 5, 6, 7, 20])
            if (any(self.PROTOCOL_TITLES[p] not in curr_suggestions for curr_suggestions in list(self.suggestions[user_id]))
                    and self.PROTOCOL_TITLES[p] not in self.recent_protocols and self.PROTOCOL_TITLES[p] not in suggestions):
                suggestions.append(self.PROTOCOL_TITLES[p])
                self.suggestions[user_id].extend([self.PROTOCOL_TITLES[p]])
        return suggestions
    ####################################################################################################

    def clear_suggestions(self, user_id):
        self.suggestions[user_id] = []
        self.reordered_protocol_questions[user_id] = deque(maxlen=5)

    def clear_emotion_scores(self, user_id):
        self.guess_emotion_predictions[user_id] = ""

    def create_new_run(self, user_id, db_session, user_session):
        new_run = UserModelRun(session_id=user_session.id)
        db_session.add(new_run)
        db_session.commit()
        self.current_run_ids[user_id] = new_run.id
        return new_run

    def clear_choices(self, user_id):
        self.user_choices[user_id] = {}

    def update_suggestions(self, user_id, protocols, app):

        # Check if user_id already has suggestions
        try:
            self.suggestions[user_id]
        except KeyError:
            self.suggestions[user_id] = []

        if type(protocols) != list:
            self.suggestions[user_id].append(deque([protocols]))
        else:
            self.suggestions[user_id].append(deque(protocols))

    # -----------------------------------------------------------------------
    def get_opening_prompt(self, user_id):
        if self.users_names[user_id] == "":
            opening_prompt = [
                "Hi, this is WillpowerBot! How are you feeling today?"]
        else:
            opening_prompt = ["Hi " + self.users_names[user_id] +
                              "! This is WillpowerBot. ", "How are you feeling today?"]
        return opening_prompt

    #     return opening_prompt
    def get_restart_prompt(self, user_id):
        if self.users_names[user_id] == "":
            restart_prompt = [
                "Please tell me again, how are you feeling today?"]
        else:
            restart_prompt = ["Please tell me again, " +
                              self.users_names[user_id] + ", how are you feeling today?"]
        return restart_prompt
#############################################################################

    def get_next_question(self, user_id):
        if self.remaining_choices[user_id] == []:
            return "project_emotion"
        else:
            selected_choice = np.random.choice(self.remaining_choices[user_id])
            self.remaining_choices[user_id].remove(selected_choice)
            return selected_choice
#############################################################################

    def add_to_reordered_protocols(self, user_id, next_protocol):
        self.reordered_protocol_questions[user_id].append(next_protocol)

    def add_to_next_protocols(self, next_protocols):
        self.protocols_to_suggest.append(deque(next_protocols))

    def clear_suggested_protocols(self):
        self.protocols_to_suggest = []

    # NOTE: this is not currently used, but can be integrated to support
    # positive protocol suggestions (to avoid recent protocols).
    # You would need to add it in when a user's emotion is positive
    # and they have chosen a protocol.

    def add_to_recent_protocols(self, recent_protocol):
        if len(self.recent_protocols) == self.recent_protocols.maxlen:
            # Removes oldest protocol
            self.recent_protocols.popleft()
        self.recent_protocols.append(recent_protocol)

    def determine_next_prompt_opening(self, user_id, app, db_session):
        user_response = self.user_choices[user_id]["choices_made"]["opening_prompt"]
        emotion = get_emotion(user_response)
        # emotion = np.random.choice(["Happy", "Sad", "Angry", "Anxious"]) #random choice to be replaced with emotion classifier
        if emotion == 'fear':
            self.guess_emotion_predictions[user_id] = 'Anxious/Scared'
            self.user_emotions[user_id] = 'Anxious'
        elif emotion == 'sadness':
            self.guess_emotion_predictions[user_id] = 'Sad'
            self.user_emotions[user_id] = 'Sad'
        elif emotion == 'anger':
            self.guess_emotion_predictions[user_id] = 'Angry'
            self.user_emotions[user_id] = 'Angry'
        else:
            self.guess_emotion_predictions[user_id] = 'Happy/Content'
            self.user_emotions[user_id] = 'Happy'
        #self.guess_emotion_predictions[user_id] = emotion
        #self.user_emotions[user_id] = emotion
        return "guess_emotion"

    def get_best_sentence(self, column, prev_qs):
        # return random.choice(column.dropna().sample(n=15).to_list()) #using random choice instead of machine learning
        maxscore = 0
        chosen = ''
        for row in column.dropna().sample(n=5):  # was 25
            fitscore = get_sentence_score(row, prev_qs)
            if fitscore > maxscore:
                maxscore = fitscore
                chosen = row
        if chosen != '':
            return chosen
        else:
            # was 25
            return random.choice(column.dropna().sample(n=5).to_list())

    def split_sentence(self, sentence):
        temp_list = re.split('(?<=[.?!]) +', sentence)
        if '' in temp_list:
            temp_list.remove('')
        temp_list = [i + " " if i[-1] in [".", "?", "!"]
                     else i for i in temp_list]
        if len(temp_list) == 2:
            return temp_list[0], temp_list[1]
        elif len(temp_list) == 3:
            return temp_list[0], temp_list[1], temp_list[2]
        else:
            return sentence


####################################################################################

    # Greeting


    def get_model_prompt_greeting(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(
        #     self.recent_questions[user_id], columns=['sentences'])
        # data = self.datasets[user_id]
        # column = data["Hello, glad you're here."].dropna()
        # my_string = self.get_best_sentence(column, prev_qs)
        # if len(self.recent_questions[user_id]) < 50:
        #     self.recent_questions[user_id].append(my_string)
        # else:
        #     self.recent_questions[user_id] = []
        #     self.recent_questions[user_id].append(my_string)
        # return self.split_sentence(my_string)
        return "intro"

    def get_model_prompt_intro(self, user_id):
        u1 = "The ability to exercise willpower is one of the most important traits of a successful leader, as well as achieving high goals in both personal and professional situations."
        u2 = "This chatbot is meant to help you strengthen your willpower based on SAT (self-attachment technique)."
        u3 = "Following SAT exercises, it gives us a chance to reraise our childhood selves. Usually, when we hear other people's advice, it's hard for us to take their advice, even when it makes sense."
        u4 = "By connecting with our childhood selves, we're offering advice to ourselves (our own childhood), so it's easier to take in these suggestions."
        u5 = "In this session, we will practice some SAT exercises. This can help you build a strong connection with your inner self and become more open to good suggestions."
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Hello, glad you're here."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return [self.split_sentence(my_string), u1, u2, u3, u4, u5]
    # Emotion Utterance

    def get_model_prompt_guess_emotion(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["All emotions - From what you have said I believe you are feeling {}. Is this correct?"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        question = my_string.format(
            self.guess_emotion_predictions[user_id].lower())
        return self.split_sentence(question)

    # User choose Emotion

    def get_model_prompt_check_emotion(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["I misunderstood you. It's my wrong. Would you mind picking one of these six options that could best describe your feeling now?"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Sad

    def get_sad_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "Sad"
        self.user_emotions[user_id] = "Sad"
        return "after_classification_negative"
    # Angry

    def get_angry_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "Angry"
        self.user_emotions[user_id] = "Angry"
        return "after_classification_negative"
    # Anxious/Scared

    def get_anxious_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "Anxious/Scared"
        self.user_emotions[user_id] = "Anxious"
        return "after_classification_negative"
    # Happy/Content

    def get_happy_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "Happy/Content"
        self.user_emotions[user_id] = "Happy"
        return "after_classification_positive"

    def get_model_prompt_project_emotion(self, user_id, app, db_session):
        if self.chosen_personas[user_id] == "Robert":
            prompt = "Ok, thank you. Now, one last important thing: since you've told me you're feeling " + self.user_emotions[user_id].lower(
            ) + ", I would like you to try to project this emotion onto your childhood self. You can press 'continue' when you are ready and I'll suggest some protocols I think may be appropriate for you."
        elif self.chosen_personas[user_id] == "Gabrielle":
            prompt = "Thank you, I will recommend some protocols for you in a moment. Before I do that, could you please try to project your " + \
                self.user_emotions[user_id].lower(
                ) + " feeling onto your childhood self? Take your time to try this, and press 'continue' when you feel ready."
        elif self.chosen_personas[user_id] == "Arman":
            prompt = "Ok, thank you for letting me know that. Before I give you some protocol suggestions, please take some time to project your current " + \
                self.user_emotions[user_id].lower(
                ) + " feeling onto your childhood self. Press 'continue' when you feel able to do it."
        elif self.chosen_personas[user_id] == "Arman":
            prompt = "Ok, thank you, I'm going to draw up a list of protocols which I think would be suitable for you today. In the meantime, going back to this " + \
                self.user_emotions[user_id].lower(
                ) + " feeling of yours, would you like to try to project it onto your childhood self? You can try now and press 'continue' when you feel ready."
        if self.chosen_personas[user_id] == "Kai":
            prompt = "Thank you. While I have a think about which protocols would be best for you, please take your time now and try to project your current " + \
                self.user_emotions[user_id].lower(
                ) + " emotion onto your childhood self. When you are able to do this, please press 'continue' to receive your suggestions."
        return self.split_sentence(prompt)
######################################################################################
    # Noble goal

    def get_model_prompt_ask_noblegoal(self, user_id, app, db_session):
        u2 = "That could be anything: get into a top university, become a successful writer, get a medal at the Olympics, etc. "
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Do you have a noble goal? When we have a noble goal, we'll be better able to mobilize our subjective initiative and our willpower will be more effectively developed."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return [self.split_sentence(my_string), u2]
    # Set Noble goal

    def get_model_prompt_set_noblegoal(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Now's the time to set your own noble goal!"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)

        ##############################################################################################

    def e1_intro(self, user_id):

        u1 = "Exercise 1:  Recalling significant early memories"
        u2 = "The first thing to do in SAT is to find out about our childhood self, let's go back in time and bring back your childhood memories. "
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Could you recall any significant early memories? Could you tell me about it? That could be positive or negative."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)

        data2 = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SAT_P) <= 6:
            u10 = self.thank_effort(user_id)
            return [u10, u2, self.split_sentence(question), self.split_sentence(question2), u1, self.u100]
        else:
            return [u2, self.split_sentence(question), self.split_sentence(question2), u1, self.u100]

    def e2_intro(self, user_id):

        u1 = "Exercise 2: Becoming intimate with our child"
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Let's try to be intimate with the child, and then they'll be glad to interact with you."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)

        data2 = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SAT_P) <= 7:
            u10 = self.thank_effort(user_id)
            return [u10, self.split_sentence(question2), u1, self.split_sentence(question), self.u100]

        else:
            return [self.split_sentence(question2), u1, self.split_sentence(question), self.u100]

    def e3_intro(self, user_id):

        u1 = "Exercise 3: Singing a song of affection"
        u3 = "There are so many beautiful songs and if you want me to choose, I would pick "
        mylist = [" Had Habits by Ed Sheeran",
                  " Break My Soul by Beyonce",
                  " As It Was by Harry Styles",
                  " Peru by Fireboy DML & Ed Sheeran",
                  " Where Are You Now by Lost Frequencies/calum Scott",
                  " Shivers by Ed Sheeran",
                  " Heat Waves by Glass Animals",
                  " Make Me Feel Good by Belters Only Ft Jazzy",
                  " Starlight by Dave",
                  " Seventeen Going Under by Sam Fender",
                  " Cold Heart by Elton John & Dua Lipa",
                  " Abcdefu by Gayle",
                  " Luude by Ft Colin Hay",
                  " Baby by Aitch/ashanti",
                  " Easy on Me by Adele",
                  " Where Did You Go by Jax Jones Ft Mnek",
                  " Surface Pressure by Jessica Darrow",
                  " First Class by Jack Harlow",
                  " Overseas by D-Block Europe Ft Central Cee",
                  " Stay by Kid Laroi & Justin Bieber",
                  " Running Up That Hill by Kate Bush",
                  " Overpass Graffiti by Ed Sheeran"
                  " the Motto by Tiesto & Ava Max",
                  " House on Fire by Mimi Webb",
                  " Anyone for You by George Ezra",
                  " Fingers Crossed by Lauren Spencer-smith",
                  " Coming for You by Switchotr Ft A1 & J1",
                  " She's All I Wanna Be by Tate Mcrae",
                  " Save Your Tears by Weeknd",
                  " Mr Brightside by Killers"]

        u4 = np.random.choice(mylist)
        u4 = u3+u4+"."
        u2 = "Don't worry, I won't judge that. Keep that secret and sing the song whenever you can.  "
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Do you have a favorite song?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)

        data2 = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SAT_P) <= 7:
            u10 = self.cong_effort(user_id)
            return [u10, self.split_sentence(question2), u1, self.split_sentence(question), u2, u4, self.u100]
        else:
            return [self.split_sentence(question2), u1, self.split_sentence(question), u2, u4, self.u100]

    def e4_intro(self, user_id):

        u1 = "Exercise 4: Expressing love and care for the child"
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Love is what kids need to grow up healthily."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)

        data2 = self.datasets[user_id]
        column2 = data["You could give the child as much love and care as you can."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SAT_P) <= 7:
            u10 = self.cong_effort(user_id)
            return [u10, self.split_sentence(question), self.split_sentence(question2), u1, self.u100]

        else:
            return [self.split_sentence(question), self.split_sentence(question2), u1, self.u100]

    def e5_intro(self, user_id):

        u1 = "Exercise 5: Pledging to care and support our child"
        u2 = "All mammals are born with an innate capacity to care for children. We're going to take care of our child like we'd take care of ourselves."
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data2 = self.datasets[user_id]
        data = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SAT_P) <= 5:
            u10 = self.thank_effort(user_id)
            return [u10, u2, self.split_sentence(question2), u1, self.u100]

        else:
            return [u2, self.split_sentence(question2), u1, self.u100]

    def e6_intro(self, user_id):

        u1 = "Exercise 6: Restoring our emotional world after our pledge"
        u2 = "Our emotional world could be re-created through art. The act of building a house is a key activity for humans and a fundamental game for children.  "
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Our goal at this point is to help our child rebuild their emotional world."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)

        data2 = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)

        data3 = self.datasets[user_id]
        column3 = data["Creating a dream house is like creating yourself."].dropna()
        question3 = self.get_best_sentence(column3, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question3)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question3)
        if len(self.suggestion_SAT_P) <= 5:
            u10 = self.cong_effort(user_id)
            return [u10, self.split_sentence(question2), u1, self.split_sentence(question), u2, self.split_sentence(question3), self.u100]

        else:
            return [self.split_sentence(question2), u1, self.split_sentence(question), u2, self.split_sentence(question3), self.u100]

    def e7_intro(self, user_id):

        u1 = "Exercise 7: Maintaining a loving relationship with the child"
        u2 = "The key to a long-lasting relationship is maintaining it. I've got some tips for keeping your relationship with your child:"
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])

        data2 = self.datasets[user_id]
        data = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SAT_P) <= 4:
            u10 = self.cong_effort(user_id)
            return [u10, u2, self.split_sentence(question2), u1, self.u100]

        else:
            return [u2, self.split_sentence(question2), u1, self.u100]

    def e8_intro(self, user_id):

        u1 = "Exercise 7b: Creating zest for life"
        u2 = "Here is a similar exercise of Exercise 7 that you could try out if you would like to go further."
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data2 = self.datasets[user_id]
        data = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SAT_P) <= 3:
            u10 = self.cong_effort(user_id)
            return [u10, self.split_sentence(question2), u1, u2, self.u100]

        else:
            return [self.split_sentence(question2), u1, u2, self.u100]

    def e9_intro(self, user_id):

        u1 = "Exercise 9: Overcoming current negative emotions"
        u2 = "If you currently have negative emotions, no worries, you're not alone! I have these little concerns too!"
        mylist = ["I once intended to go to the party but my mom refused to let me hang out. So I did not make it and my friends thought I didn't consider their feelings. In those days, being misunderstood by my close friends really embarrassed me.",
                  "It has been raining for many days, and I cannot go out to play. This embarasses me a lot.",
                  "During this hot summer, I don't have an air conditioner in my room, and that's horrible.",
                  "I felt uncomfortable when the project I have to complete is very complex.",
                  "I sometimes get angry with my friends for them being late.",
                  "When other people use my stuff without my permission, that's quite annoying",
                  "Once when we went to a party, the friend who invited me and was supposed to go with me decided not to go. This is quite annoying.",
                  "When I did not do anything but someone said I did, I got upset.",
                  "When I cared someone a lot but cannot receive his/her response, I really hate it."
                  ]
        u3 = np.random.choice(mylist)
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        data2 = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SAT_P) <= 5:
            u10 = self.cong_effort(user_id)
            return [u10, u2, u3, self.split_sentence(question2), u1, self.u100]

        else:
            return [u2, u3, self.split_sentence(question2), u1, self.u100]

    def e10_intro(self, user_id):

        u1 = "Exercise 10: Overcoming past pain"
        u2 = "The first thing to do in SAT is to find out about our childhood self, let's go back in time and bring back your childhood memories. "
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Could you recall any significant early memories? Could you tell me about it? That could be positive or negative."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)

        data2 = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SAT_P) <= 5:
            u10 = self.thank_effort(user_id)
            return [u10, self.split_sentence(question2), u1, u2, self.split_sentence(question), self.u100]
        else:
            return [self.split_sentence(question2), u1, u2, self.split_sentence(question), self.u100]

    def e11_intro(self, user_id):

        u1 = "Exercise 11: Muscle relaxation and playful face"
        u2 = "By releasing dopamine and serotonin, laughter helps fight anxiety and depression. Here's what I suggest:"
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Smiling and laughing are essential to reraise our childhood selves."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        if len(self.suggestion_SAT_P) <= 5:
            u10 = self.thank_effort(user_id)
            return [u10, self.split_sentence(question), u2, u1, self.u100]

        else:
            return [self.split_sentence(question), u2, u1, self.u100]

    def e12_intro(self, user_id):

        u1 = "Exercise 12: Laughing on our own"
        u2 = "Would you be able to recall something you have accomplished recently, such as helping a stranger find the way, finishing a book, or going to the gym?"
        mylist = ["I did chores today.", "I talked to my neighbor today.", "I finished reading a book this month.", "I got good marks for my exams.", "I accomplished to go to gym 5 days each week!",
                  "I performed a presentation of my work. ", "I helped filling out project survey of my friends XD.", "I cooked a meal and my friends really enjoyed it!", "I have finished following a TV series.",
                  "I did research on topics I am interested in!", "I completed two projects in my work!", "I recently have achieved a good average marks for my exam!", "Today when I got out of the gym and went into the chaging room, a person asked me how to lock the cabinet. I told him it needed one pound to lock it and gave him one pound. And then we add each other's contact number."
                  ]
        u3 = np.random.choice(mylist)
        u3 = "For example "+u3
        u4 = "Smile at your accomplishment when you're comfortable, then laugh at it! "
        if len(self.suggestion_SAT_P) <= 5:
            u10 = self.thank_effort(user_id)
            return [u10, u2, u3, u4, u1]
        else:
            return [u2, u3, u4, u1]

    def e14_intro(self, user_id):

        u1 = "Exercise 14: Creating your own brand of laughter"
        u2 = "Of course, I don't want you to laugh like them, I mean your special laugh can become your brand, into muscle memory, so you subconsciously smile often and can control how much you smile, not feel embarrassed."
        u3 = "Are you aware that there are thousands of different forms of laughter? "
        u4 = "The laughter of various characters in One Piece manga series is different. I believe that normal people will not laugh like this, but it does not prevent all kinds of weird laughter from becoming one of the classic symbols of One Piece."
        u5 = "Isn't it fun? Come and try it!"
        if len(self.suggestion_SAT_P) <= 5:
            u10 = self.cong_effort(user_id)
            return [u10, u3, u4, u2, u5, u1, self.u100]
        else:
            return [u3, u4, u2, u5, u1, self.u100]

    def e15_intro(self, user_id):

        u1 = "Exercise 15: Learning to change our perspective"
        u2 = "And sometimes hardships become wealth later on."
        mylist = ["I can learn a lot from challenging projects.",
                  "the concert was canceled, but I had more time to study.",
                  "there is no air conditioner inside of my room and this is really a bad news for me and made me upset in the very beginning. Then, I told myself that I can leave the room and find a place with air conditioner so that I can study under a peaceful mood. This is the way I cheer myself up.",
                  "when I'm sick, I tell myself I can boost my immunity haha.",
                  "when I fail one experiment, I can exclude the wrong option at least.",
                  "days ago, I could not finish modelling the wind farm system, one of goals of my final project. And then I told myself, at least I had read all the paper I need to finish the task and the only thing remained to be done is to be patient and confident."
                  ]
        u3 = np.random.choice(mylist)
        u3 = "For example "+u3
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["You could usually come up with better ideas if you think from different perspectives."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)

        data2 = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SAT_P) <= 5:
            u10 = self.cong_effort(user_id)
            return [u10, self.split_sentence(question2), u1, u2, u3, self.split_sentence(question)]
        else:
            return [self.split_sentence(question2), u1, u2, u3, self.split_sentence(question)]

    def e16_intro(self, user_id):

        u1 = "Exercise 16: Learning to be playful about our past pains"
        u2 = "The present and the future are the only things we can change. So let me help you get out of the pain as soon as possible! "
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Are there any past pains you want to get over? We could tackle these traumas through some exercise."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)

        data2 = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SAT_P) <= 3:
            u10 = self.cong_effort(user_id)
            return [u10, self.split_sentence(question), u2, self.split_sentence(question2), u1]

        else:
            return [self.split_sentence(question), u2, self.split_sentence(question2), u1]

    def e20_intro(self, user_id):

        u1 = "Exercise 20: Practicing Affirmations"
        u2 = "Are you familiar with Nietzsche or Laozi? Both are famous philosopher. Here's what they said:"
        mylist = [" 'What does not kill me makes me stronger.' — Friedrich Nietzsche's Twilight of the Idols (1888)",
                  " 'A journey of a thousand miles begins with a single step.'  —Chapter 64 of the Dao De Jing ascribed to Laozi",
                  " 'My formula for greatness in a human being is Amor fati: that one wants nothing to be different, not forward, not backward, not in all eternity. Not merely bear what is necessary, still less conceal it—all idealism is mendacity in the face of what is necessary—but love it.'  — Friedrich Nietzsche, 1888",
                  " 'To those human beings who are of any concern to me I wish suffering, desolation, sickness, ill-treatment, indignities—I wish that they should not remain unfamiliar with profound self-contempt, the torture of self-mistrust, the wretchedness of the vanquished: I have no pity for them, because I wish them the only thing that can prove today whether one is worth anything or not—that one endures.'  ― Friedrich Nietzsche, The Will to Power",
                  " 'Whoever fights monsters should see to it that in the process he does not become a monster. And if you gaze long enough into an abyss, the abyss will gaze back into you.' ― Friedrich Nietzsche"
                  ]
        u3 = np.random.choice(mylist)
        u4 = "You may find useful to read these affirmations."
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data2 = self.datasets[user_id]
        data = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SAT_P) <= 6:
            u10 = self.cong_effort(user_id)
            return [u10, u2, u3, u4, self.split_sentence(question2), u1]
        else:
            return [u2, u3, u4, self.split_sentence(question2), u1]

    def e20_intro(self, user_id):

        u1 = "Exercise 20: Practicing Affirmations"
        u2 = "Are you familiar with Nietzsche or Laozi? Both are famous philosopher. Here's what they said:"
        mylist = [" 'What does not kill me makes me stronger.' — Friedrich Nietzsche's Twilight of the Idols (1888)",
                  " 'A journey of a thousand miles begins with a single step.'  —Chapter 64 of the Dao De Jing ascribed to Laozi",
                  " 'My formula for greatness in a human being is Amor fati: that one wants nothing to be different, not forward, not backward, not in all eternity. Not merely bear what is necessary, still less conceal it—all idealism is mendacity in the face of what is necessary—but love it.'  — Friedrich Nietzsche, 1888",
                  " 'To those human beings who are of any concern to me I wish suffering, desolation, sickness, ill-treatment, indignities—I wish that they should not remain unfamiliar with profound self-contempt, the torture of self-mistrust, the wretchedness of the vanquished: I have no pity for them, because I wish them the only thing that can prove today whether one is worth anything or not—that one endures.'  ― Friedrich Nietzsche, The Will to Power",
                  " 'Whoever fights monsters should see to it that in the process he does not become a monster. And if you gaze long enough into an abyss, the abyss will gaze back into you.' ― Friedrich Nietzsche"
                  ]
        u3 = np.random.choice(mylist)
        u4 = "You may find useful to read these affirmations."
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data2 = self.datasets[user_id]
        data = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SAT_P) <= 6:
            u10 = self.cong_effort(user_id)
            return [u10, u2, u3, u4, self.split_sentence(question2), u1]
        else:
            return [u2, u3, u4, self.split_sentence(question2), u1]

    def e20_intros(self, user_id):

        u1 = "Exercise 20: Practicing Affirmations"
        u2 = "Are you familiar with Nietzsche or Laozi? Both are famous philosopher. Here's what they said:"
        mylist = [" 'What does not kill me makes me stronger.' — Friedrich Nietzsche's Twilight of the Idols (1888)",
                  " 'A journey of a thousand miles begins with a single step.'  —Chapter 64 of the Dao De Jing ascribed to Laozi",
                  " 'My formula for greatness in a human being is Amor fati: that one wants nothing to be different, not forward, not backward, not in all eternity. Not merely bear what is necessary, still less conceal it—all idealism is mendacity in the face of what is necessary—but love it.'  — Friedrich Nietzsche, 1888",
                  " 'To those human beings who are of any concern to me I wish suffering, desolation, sickness, ill-treatment, indignities—I wish that they should not remain unfamiliar with profound self-contempt, the torture of self-mistrust, the wretchedness of the vanquished: I have no pity for them, because I wish them the only thing that can prove today whether one is worth anything or not—that one endures.'  ― Friedrich Nietzsche, The Will to Power",
                  " 'Whoever fights monsters should see to it that in the process he does not become a monster. And if you gaze long enough into an abyss, the abyss will gaze back into you.' ― Friedrich Nietzsche"
                  ]
        u3 = np.random.choice(mylist)
        u4 = "You may find useful to read these affirmations."
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data2 = self.datasets[user_id]
        data = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SPE) <= 6:
            u10 = self.cong_effort(user_id)
            return [u10, u2, u3, u4, self.split_sentence(question2), u1]
        else:
            return [u2, u3, u4, self.split_sentence(question2), u1]

    def e21_intro(self, user_id):

        u1 = "Exercise 21: Limit your materialistic desires and sensual pleasure"
        u2 = "Do you think you indulge in overeating, alcohol, soft drugs, cigarettes, or sensual pleasures?"
        u3 = "Bad habits always influence you in your daily life, so don't underestimate their impact."
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Your life could be much better if you fix these bad habits."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)

        data2 = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SPE) <= 3:
            u10 = self.thank_effort(user_id)
            return [u10, u2, u3, self.split_sentence(question2), u1, self.split_sentence(question), self.u100]
        else:
            return [u2, u3, self.split_sentence(question2), u1, self.split_sentence(question), self.u100]

    def e22_intro(self, user_id):

        u1 = "Exercise 22 Forgo an instance of your bad habits every day"
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Your life could be much better if you fix these bad habits."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        data2 = self.datasets[user_id]
        column2 = data["Here is a similar one that you could practice to reinforce your willpower in your daily life:"].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SPE) <= 3:
            u10 = self.thank_effort(user_id)
            return [u10, self.split_sentence(question2), u1, self.split_sentence(question), self.u100]
        else:
            return [self.split_sentence(question2), u1, self.split_sentence(question), self.u100]

    def e23_intro(self, user_id):

        u1 = "Exercise 23 Strengthen your muscle to strengthen your willpower"
        u3 = "Are you a frequent gym user and do you exercise a lot?"
        u2 = "Muscle strength is related to willpower strength. You could strengthen your willpower by enduring minor physical pain during exercising or in everyday life."
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["It sounded crazy to me too at first, but there have been studies on it."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)

        data2 = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SPE) <= 3:
            u10 = self.cong_effort(user_id)
            return [u10, u3, u2, self.split_sentence(question),  self.split_sentence(question2), u1, self.u100]
        else:
            return [u3, u2, self.split_sentence(question),  self.split_sentence(question2), u1, self.u100]

    def e24_intro(self, user_id):

        u1 = "24 Need less, lead a simple life"
        u2 = "Are you tempted to buy stuff when you see ads on videos or on the street? Are you tempted to buy the latest mobile phone, or wear the most popular clothes when you see your friends/colleagues having them? "
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["It's crazy how powerful adverts and peer pressure can be. We're more consumerist now thanks to ads and peer pressure."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)

        data2 = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        if len(self.suggestion_SPE) <= 2:
            u10 = self.cong_effort(user_id)
            return [u10, u2, self.split_sentence(question), self.split_sentence(question2), u1, self.u100]
        else:
            return [u2, self.split_sentence(question), self.split_sentence(question2), u1, self.u100]

    def thank_effort(self, user_id):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Thank you for your efforts until now."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def cong_effort(self, user_id):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Congratulations, you are now getting closer to have a strong willpower."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    # Negative Reassure

    def get_model_prompt_negative_reassure(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Don't worry. I'm right here to help you deal with this negative emotion in a more constructive way."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)

    def get_model_prompt_sat_or_aff(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Here are 2 plans to help enhance your willpower, you could choose your favorite one."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        u2 = "Would you like to do SAT Exercises or try Exercise 25 Read Affirmations?"
        return [self.split_sentence(my_string), u2]

    def get_model_prompt_Read_Affirmations(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data[""].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)

    def get_model_prompt_saviour(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + \
            " - Do you believe that you should be the saviour of someone else?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_victim(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + \
            " - Do you see yourself as the victim, blaming someone else for how negative you feel?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_controlling(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + \
            " - Do you feel that you are trying to control someone?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_accusing(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + \
            " - Are you always blaming and accusing yourself for when something goes wrong?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    # Specific reasons/events Bad mood

    def get_model_prompt_specific_event(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = "May I ask if there are any specific reasons/events that caused your bad mood?"
        base_prompt2 = "Don't worry. I'm right here to help you deal with this negative emotion in a more constructive way."
        # base_prompt = self.user_emotions[user_id] + \
        #     " - Was this caused by a specific event/s?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)

        column2 = data[base_prompt2].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)

        return [self.split_sentence(question2), self.split_sentence(question)]
    # Recent event

    def get_model_prompt_event_is_recent(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = "Was this caused by a recent or distant event? "
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)
    # Past pain

    def get_model_prompt_past_pain(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Are there any past pains you want to get over? We could tackle these traumas through some exercise."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Different Perspective

    def get_model_prompt_diff_pers(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["You could usually come up with better ideas if you think from different perspectives."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Project emotion

    def get_model_projectemotion(self, user_id):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["I would like you to imagine yourself as your childhood self and project negative emotions onto him or her."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Early memories

    def get_model_early_memo(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Could you recall any significant early memories? Could you tell me about it? That could be positive or negative."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)

    # Fix bad habit

    def get_model_fix_bad_habits(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Your life could be much better if you fix these bad habits."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Peer pressure

    def get_model_peer_pressure(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["It's crazy how powerful adverts and peer pressure can be. We're more consumerist now thanks to ads and peer pressure."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Additional Question

    def get_model_add_q(self, user_id):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Is it ok for me to ask additional questions? I will be able to gain a better understanding of your situation this way."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Another ex

    def get_model_another_ex(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Would you like to try another exercise to help strengthen your willpower? "].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Another session

    def get_model_another_session(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Would you like to restart for another session?"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Suggest exercise

    def get_model_suggest_ex(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["I want to suggest you this exercise."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Suggest many exercise

    def get_model_suggest_exs(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["I'd like to suggest a few exercises to you."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Recommand

    def get_model_recommand_ex(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["You could give it a try."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Choose path

    # def get_model_choose_path(self, user_id, app, db_session):
    #     prev_qs = pd.DataFrame(
    #         self.recent_questions[user_id], columns=['sentences'])
    #     data = self.datasets[user_id]
    #     column = data["Here are 2 plans to help enhance your willpower, you could choose your favourite one."].dropna()
    #     my_string = self.get_best_sentence(column, prev_qs)
    #     if len(self.recent_questions[user_id]) < 50:
    #         self.recent_questions[user_id].append(my_string)
    #     else:
    #         self.recent_questions[user_id] = []
    #         self.recent_questions[user_id].append(my_string)
    #     return self.split_sentence(my_string)
    # Choose path

    def get_model_choose_path(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Here are 2 plans to help enhance your willpower, you could choose your favourite one."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # New ex

    def get_model_try_newex(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Are you ready to try another new exercise?"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Similar ex

    def get_model_similar_ex(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Here is an exercise that you could practice to reinforce your willpower in your daily life."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Crazy

    def get_model_crazy(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["It sounded crazy to me too at first, but there have been studies on it."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Encourage

    def get_model_encourage(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Strengthening your willpower is not a piece of cake."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Thank u

    def get_model_thank_u(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Thank you for your efforts until now."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Congra

    def get_model_congra(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Congratulations, you are now getting closer to have a strong willpower."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Thank u answer

    def get_model_thank_u_answer(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Thank you for answering my question."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Ex end

    def get_model_ex_end(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Amazing! You have tried out all the exercises that I could come up with."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Intimate

    def get_model_ex_end(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Let's try to be intimate with the child, and then they'll be glad to interact with you."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Love

    def get_model_love(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Love is what kids need to grow up healthily."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Smile Laugh

    def get_model_smile_laugh(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Smiling and laughing are essential to reraise our childhood selves."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Rebuild emotional world

    def get_model_love(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Our goal at this point is to help our child rebuild their emotional world. "].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Love and care

    def get_model_love_care(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["You could give the child as much love and care as you can."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
    # Dream house

    def get_model_love(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Creating a dream house is like creating yourself."].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)

    def get_model_prompt_revisit_recent(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = "Have you recently attempted Exercise 9: Overcoming Your Current Negative Emotions? And if so did it trigger uncomfortable?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def distressing(self, user_id):
        u1 = ["OK, I see. Then I would recommend this exercise:"]
        return "e8_nr"

    def not_distressing(self, user_id):
        u1 = ["Good, then you could practice this exercise as often as you like to overcome your current negative emotions."]
        u2 = "To continue, I would recommend this exercise:"
        return [u1, u2, "e8_nr"]

    def get_model_prompt_revisit_distant(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = "Have you recently attempted Exercise 10: Overcoming Past Pain? And if so did it trigger uncomfortable?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    # Add question
    def get_model_prompt_more_questions(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = "Is it ok for me to ask additional questions? I will be able to gain a better understanding of your situation this way."
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_antisocial(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + \
            " - Have you strongly felt or expressed any of the following emotions towards someone:"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return [self.split_sentence(question), "Envy, jealousy, greed, hatred, mistrust, malevolence, or revengefulness?"]

    def get_model_prompt_rigid_thought(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + \
            " - In previous conversations, have you considered other viewpoints presented?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_personal_crisis(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + \
            " - Are you undergoing a personal crisis (experiencing difficulties with loved ones e.g. falling out with friends)?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_happy(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Happy - That's Good! Let me recommend a protocol you can attempt."].dropna()
        u2 = "Here are 2 plans to help enhance your willpower, you could choose your favorite one."
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return [self.split_sentence(question), u2]

    def get_model_prompt_suggestions(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["All emotions - Here are my recommendations, please select the protocol that you would like to attempt"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_trying_protocol(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["All emotions - Please try to go through this protocol now. When you finish, press 'continue'"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        # print(self.current_protocol_ids[user_id][0])
        return self.split_sentence(question)
        # return ["You have selected Protocol " + str(self.current_protocol_ids[user_id][0]) + ". ", self.split_sentence(question)]

    def get_model_prompt_try_protocol(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["All emotions - Please try to go through this protocol now. When you finish, press 'continue'"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        # print(self.current_protocol_ids[user_id][0])
        return self.split_sentence(question)
        # return ["You have selected Protocol " + str(self.current_protocol_ids[user_id][0]) + ". ", self.split_sentence(question)]

    def get_model_prompt_found_useful(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["All emotions - Do you feel better or worse after having taken this protocol?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        u1 = "Go ahead with the protocol in your own time. "
        return self.split_sentence(question)

    def get_model_prompt_new_better(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["All emotions - Would you like to attempt another protocol? (Patient feels better)"].dropna(
        )
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_new_worse(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["All emotions - Would you like to attempt another protocol? (Patient feels worse)"].dropna(
        )
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_ending(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["All emotions - Thank you for taking part. See you soon"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return [self.split_sentence(question), "You have been disconnected. Refresh the page if you would like to start over."]

    def determine_next_prompt_new_protocol(self, user_id, app):
        try:
            self.suggestions[user_id]
        except KeyError:
            self.suggestions[user_id] = []
        if len(self.suggestions[user_id]) > 0:
            return "suggestions"
        return "more_questions"

    def determine_positive_protocols(self, user_id, app):
        protocol_counts = {}
        total_count = 0

        for protocol in self.positive_protocols:
            count = Protocol.query.filter_by(protocol_chosen=protocol).count()
            protocol_counts[protocol] = count
            total_count += count

        # for protocol in counts:
        if total_count > 10:
            first_item = min(zip(protocol_counts.values(),
                             protocol_counts.keys()))[1]
            del protocol_counts[first_item]

            second_item = min(zip(protocol_counts.values(),
                              protocol_counts.keys()))[1]
            del protocol_counts[second_item]

            third_item = min(zip(protocol_counts.values(),
                             protocol_counts.keys()))[1]
            del protocol_counts[third_item]
        else:
            # CASE: < 10 protocols undertaken in total, so randomness introduced
            # to avoid lowest 3 being recommended repeatedly.
            # Gives number of next protocol to be suggested
            first_item = np.random.choice(
                list(set(self.positive_protocols) - set(self.recent_protocols))
            )
            second_item = np.random.choice(
                list(
                    set(self.positive_protocols)
                    - set(self.recent_protocols)
                    - set([first_item])
                )
            )
            third_item = np.random.choice(
                list(
                    set(self.positive_protocols)
                    - set(self.recent_protocols)
                    - set([first_item, second_item])
                )
            )

        return [
            self.PROTOCOL_TITLES[first_item],
            self.PROTOCOL_TITLES[second_item],
            self.PROTOCOL_TITLES[third_item],
        ]

    def determine_specific_ex(self, user_id, app):
        protocol_counts = {}
        total_count = 0

        for protocol in self.specific_ex:
            count = Protocol.query.filter_by(protocol_chosen=protocol).count()
            protocol_counts[protocol] = count
            total_count += count

        # for protocol in counts:
        if total_count > 10:
            first_item = min(zip(protocol_counts.values(),
                             protocol_counts.keys()))[1]
            del protocol_counts[first_item]

            second_item = min(zip(protocol_counts.values(),
                              protocol_counts.keys()))[1]
            del protocol_counts[second_item]

            third_item = min(zip(protocol_counts.values(),
                             protocol_counts.keys()))[1]
            del protocol_counts[third_item]
        else:
            # CASE: < 10 protocols undertaken in total, so randomness introduced
            # to avoid lowest 3 being recommended repeatedly.
            # Gives number of next protocol to be suggested
            first_item = np.random.choice(
                list(set(self.specific_ex) - set(self.recent_protocols))
            )
            second_item = np.random.choice(
                list(
                    set(self.specific_ex)
                    - set(self.recent_protocols)
                    - set([first_item])
                )
            )
            third_item = np.random.choice(
                list(
                    set(self.specific_ex)
                    - set(self.recent_protocols)
                    - set([first_item, second_item])
                )
            )

        return [
            self.PROTOCOL_TITLES[first_item],
            self.PROTOCOL_TITLES[second_item],
            self.PROTOCOL_TITLES[third_item],
        ]

    def determine_protocols_keyword_classifiers(
        self, user_id, db_session, curr_session, app
    ):

        # We add "suggestions" first, and in the event there are any left over we use those, otherwise we divert past it.
        self.add_to_reordered_protocols(user_id, "suggestions")

        # Default case: user should review protocols 13 and 14.
        #self.add_to_next_protocols([self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[14]])
        return self.get_next_protocol_question(user_id, app)

    def update_conversation(self, user_id, new_dialogue, db_session, app):
        try:
            session_id = self.user_choices[user_id]["current_session_id"]
            curr_session = UserModelSession.query.filter_by(
                id=session_id).first()
            if curr_session.conversation is None:
                curr_session.conversation = "" + new_dialogue
            else:
                curr_session.conversation = curr_session.conversation + new_dialogue
            curr_session.last_updated = datetime.datetime.utcnow()
            db_session.commit()
        except KeyError:
            curr_session = UserModelSession(
                user_id=user_id,
                conversation=new_dialogue,
                last_updated=datetime.datetime.utcnow(),
            )

            db_session.add(curr_session)
            db_session.commit()
            self.user_choices[user_id]["current_session_id"] = curr_session.id

    def save_current_choice(
        self, user_id, input_type, user_choice, user_session, db_session, app
    ):
        # Set up dictionary if not set up already
        # with Session() as session:

        try:
            self.user_choices[user_id]
        except KeyError:
            self.user_choices[user_id] = {}

        # Define default choice if not already set
        try:
            current_choice = self.user_choices[user_id]["choices_made"][
                "current_choice"
            ]
        except KeyError:
            current_choice = self.QUESTION_KEYS[0]

        try:
            self.user_choices[user_id]["choices_made"]
        except KeyError:
            self.user_choices[user_id]["choices_made"] = {}

        if current_choice == "ask_name":
            self.clear_suggestions(user_id)
            self.user_choices[user_id]["choices_made"] = {}
            self.create_new_run(user_id, db_session, user_session)

        # Save current choice
        self.user_choices[user_id]["choices_made"]["current_choice"] = current_choice
        self.user_choices[user_id]["choices_made"][current_choice] = user_choice

        curr_prompt = self.QUESTIONS[current_choice]["model_prompt"]
        # prompt_to_use = curr_prompt
        if callable(curr_prompt):
            curr_prompt = curr_prompt(user_id, db_session, user_session, app)

        # removed stuff here

        else:
            self.update_conversation(
                user_id,
                "Model:{} \nUser:{} \n".format(curr_prompt, user_choice),
                db_session,
                app,
            )

        # Case: update suggestions for next attempt by removing relevant one
        if (
            current_choice == "suggestions"
        ):

            # PRE: user_choice is a string representing a number from 1-20,
            # or the title for the corresponding protocol

            try:
                current_protocol = self.TITLE_TO_PROTOCOL[user_choice]
            except KeyError:
                current_protocol = int(user_choice)

            protocol_chosen = Protocol(
                protocol_chosen=current_protocol,
                user_id=user_id,
                session_id=user_session.id,
                run_id=self.current_run_ids[user_id],
            )
            db_session.add(protocol_chosen)
            db_session.commit()
            self.current_protocol_ids[user_id] = [
                current_protocol, protocol_chosen.id]

            for i in range(len(self.suggestions[user_id])):
                curr_protocols = self.suggestions[user_id][i]
                if curr_protocols[0] == self.PROTOCOL_TITLES[current_protocol]:
                    curr_protocols.popleft()
                    if len(curr_protocols) == 0:
                        self.suggestions[user_id].pop(i)
                    break

        # PRE: User choice is string in ["Better", "Worse"]
        elif current_choice == "user_found_useful":
            current_protocol = Protocol.query.filter_by(
                id=self.current_protocol_ids[user_id][1]
            ).first()
            current_protocol.protocol_was_useful = user_choice
            db_session.commit()

        if current_choice == "guess_emotion":
            option_chosen = user_choice + " ({})".format(
                self.guess_emotion_predictions[user_id]
            )
        else:
            option_chosen = user_choice
        choice_made = Choice(
            choice_desc=current_choice,
            option_chosen=option_chosen,
            user_id=user_id,
            session_id=user_session.id,
            run_id=self.current_run_ids[user_id],
        )
        db_session.add(choice_made)
        db_session.commit()

        return choice_made

    def determine_next_choice(
        self, user_id, input_type, user_choice, db_session, user_session, app
    ):
        # Find relevant user info by using user_id as key in dict.
        #
        # Then using the current choice and user input, we determine what the next
        # choice is and return this as the output.

        # Some edge cases to consider based on the different types of each field:
        # May need to return list of model responses. For next protocol, may need
        # to call function if callable.

        # If we cannot find the specific choice (or if None etc.) can set user_choice
        # to "any".

        # PRE: Will be defined by save_current_choice if it did not already exist.
        # (so cannot be None)

        current_choice = self.user_choices[user_id]["choices_made"]["current_choice"]
        current_choice_for_question = self.QUESTIONS[current_choice]["choices"]
        current_protocols = self.QUESTIONS[current_choice]["protocols"]
        if input_type != "open_text":
            if (
                current_choice != "suggestions"
                and current_choice != "suggestions_will"
                and current_choice != "event_is_recent"
                and current_choice != "more_questions"
                and current_choice != "after_classification_positive"
                and current_choice != "user_found_useful_e1"
                and current_choice != "check_emotion"
                and current_choice != "user_found_useful_e21"
                and current_choice != "user_found_useful_e1"
                and current_choice != "user_found_useful_e31"
                and current_choice != "user_found_useful_e41"
                and current_choice != "new_protocol_better_e21"
                and current_choice != "new_protocol_worse_e21"
                and current_choice != "new_protocol_same_e21"
                and current_choice != "new_protocol_better_e1"
                and current_choice != "new_protocol_worse_e1"
                and current_choice != "new_protocol_same_e1"
                and current_choice != "new_protocol_better_e31"
                and current_choice != "new_protocol_worse_e31"
                and current_choice != "new_protocol_same_e31"
                and current_choice != "new_protocol_better_e41"
                and current_choice != "new_protocol_worse_e41"
                and current_choice != "new_protocol_same_e41"
                and current_choice != "choose_persona"
                and current_choice != "project_emotion"
                and current_choice != "after_classification_negative"
                and current_choice != "ask_noble_goal"
                and current_choice != "greeting"
                and current_choice != "sat_or_25"
                and current_choice != "favorit_song"
                and current_choice != "trying_protocol"
                and current_choice != "no_more_choice"
                and current_choice != "no_more_choice_spe"
                and current_choice != "no_more_choice_nr"
                and current_choice != "no_more_choice_nd"
                and current_choice != "suggestions_P_SAT"
                and current_choice != "e1"
                and current_choice != "e2"
                and current_choice != "e3"
                and current_choice != "e4"
                and current_choice != "e5"
                and current_choice != "e6"
                and current_choice != "e7"
                and current_choice != "e8"
                and current_choice != "e9"
                and current_choice != "e10"
                and current_choice != "e11"
                and current_choice != "e12"
                and current_choice != "e14"
                and current_choice != "e15"
                and current_choice != "e16"
                and current_choice != "e20"
                and current_choice != "e21"
                and current_choice != "e22"
                and current_choice != "e23"
                and current_choice != "e24"
                and current_choice != "e24s"
                and current_choice != "e20s"
                and current_choice != "e23s"
                and current_choice != "e22s"
                and current_choice != "e21s"
                and current_choice != "e9_comfort"
                and current_choice != "e9_nr"
                and current_choice != "e8_nr"
                and current_choice != "e11_nr"
                and current_choice != "add_q"
                and current_choice != "project_e15"
                and current_choice != "new_protocol_same_e15"
                and current_choice != "new_protocol_worse_e15"
                and current_choice != "new_protocol_better_e15"
                and current_choice != "user_found_useful_e15_re"
                and current_choice != "e15_reluctant"
                and current_choice != "e11_nr"
                and current_choice != "qa"

                and current_choice != "e16_nd"
                and current_choice != "e15_nd"
                and current_choice != "e10_comfort"
                and current_choice != "e10_nd"
                and current_choice != "new_protocol_better_e61"
                and current_choice != "new_protocol_same_e51"
                and current_choice != "new_protocol_worse_e51"
                and current_choice != "new_protocol_better_e51"
                and current_choice != "user_found_useful_e51"
                and current_choice != "user_found_useful_e61"
                and current_choice != "new_protocol_worse_e61"
                and current_choice != "new_protocol_same_e61"





            ):
                user_choice = user_choice.lower()

            if (
                current_choice == "suggestions"
            ):
                try:
                    current_protocol = self.TITLE_TO_PROTOCOL[user_choice]
                except KeyError:
                    current_protocol = int(user_choice)
                protocol_choice = self.PROTOCOL_TITLES[current_protocol]
                next_choice = current_choice_for_question[protocol_choice]
                protocols_chosen = current_protocols[protocol_choice]

            elif (
                current_choice == "suggestions_P_SAT"
            ):
                try:
                    current_protocol = self.TITLE_TO_PROTOCOL[user_choice]
                except KeyError:
                    current_protocol = int(user_choice)
                protocol_choice = self.PROTOCOL_TITLES[current_protocol]

                next_choice = current_choice_for_question[protocol_choice]
                protocols_chosen = current_protocols[protocol_choice]

            elif current_choice == "check_emotion":
                if user_choice == "Sad":
                    next_choice = current_choice_for_question["Sad"]
                    protocols_chosen = current_protocols["Sad"]
                elif user_choice == "Angry":
                    next_choice = current_choice_for_question["Angry"]
                    protocols_chosen = current_protocols["Angry"]
                elif user_choice == "Anxious/Scared":
                    next_choice = current_choice_for_question["Anxious/Scared"]
                    protocols_chosen = current_protocols["Anxious/Scared"]
                else:
                    next_choice = current_choice_for_question["Happy/Content"]
                    protocols_chosen = current_protocols["Happy/Content"]
            else:
                next_choice = current_choice_for_question[user_choice]
                protocols_chosen = current_protocols[user_choice]

        else:
            next_choice = current_choice_for_question["open_text"]
            protocols_chosen = current_protocols["open_text"]

        if callable(next_choice):
            next_choice = next_choice(user_id, db_session, user_session, app)

        if current_choice == "guess_emotion" and user_choice.lower() == "yes":
            if self.guess_emotion_predictions[user_id] == "Sad":
                next_choice = next_choice["Sad"]
            elif self.guess_emotion_predictions[user_id] == "Angry":
                next_choice = next_choice["Angry"]
            elif self.guess_emotion_predictions[user_id] == "Anxious/Scared":
                next_choice = next_choice["Anxious/Scared"]
            else:
                next_choice = next_choice["Happy/Content"]

        if callable(protocols_chosen):
            protocols_chosen = protocols_chosen(
                user_id, db_session, user_session, app)
        next_prompt = self.QUESTIONS[next_choice]["model_prompt"]
        if callable(next_prompt):
            next_prompt = next_prompt(user_id, db_session, user_session, app)
        if (
            len(protocols_chosen) > 0
            and current_choice != "suggestions"
        ):
            self.update_suggestions(user_id, protocols_chosen, app)

        # Case: new suggestions being created after first protocol attempted
        if next_choice == "opening_prompt":
            self.clear_suggestions(user_id)
            self.clear_emotion_scores(user_id)
            self.create_new_run(user_id, db_session, user_session)

        if next_choice == "suggestions":
            next_choices = self.get_suggestions(user_id, app)

        # if next_choice == "suggestions_P_SAT":
        #     next_choices = self.get_new_suggestions(user_id, app)
        # if next_choice == "suggestions_N_SAT_R":
        #     next_choices = self.get_suggestions_N_SAT_R(user_id, app)
        # if next_choice == "suggestions_N_SAT_D":
        #     next_choices = self.get_suggestions_N_SAT_D(user_id, app)
        else:
            next_choices = list(self.QUESTIONS[next_choice]["choices"].keys())
        self.user_choices[user_id]["choices_made"]["current_choice"] = next_choice
        return {"model_prompt": next_prompt, "choices": next_choices}
