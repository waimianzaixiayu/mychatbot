import numpy as np
suggestion_SAT_P = [1, 2, 3, 4, 5, 6, 7, 20, 9, 12, 14]


def get_new_suggestions(self, user_id):
    if suggestion_SAT_P == []:
        return "no_more_choice"
    else:
        selected_choice = np.random.choice(suggestion_SAT_P)
        suggestion_SAT_P.remove(selected_choice)
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

            "e1": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e1_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e2": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e2_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e3": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e3_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e4": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e4_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e5": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e5_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },

            "e6": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e6_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e7": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e7_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e8": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e8_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e9": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e9_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e10": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e10_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e11": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e11_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e12": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e12_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e14": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e14_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e15": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e15_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e16": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e16_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e20": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e20_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e21": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e21_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e22": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e22_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e23": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e23_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e24": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e24_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },
            "e25": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.e25_intro(user_id),
                "choices": {
                    "Yes, I want follow up exercises": lambda user_id, db_session, curr_session, app: self.get_new_suggestions(user_id),
                    "No, end session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I want follow up exercises": [],
                    "No, end session": []
                },
            },

    def e1_intro(self, user_id, app, db_session):
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
        return [self.split_sentence(question2), u1, u2, self.split_sentence(question)]

    def e2_intro(self, user_id, app, db_session):
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
        return [self.split_sentence(question2), u1, self.split_sentence(question)]

    def e3_intro(self, user_id, app, db_session):
        u1 = "Exercise 3: Singing a song of affection"
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
        return [self.split_sentence(question2), u1, self.split_sentence(question), u2]

    def e4_intro(self, user_id, app, db_session):
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
        return [self.split_sentence(question), self.split_sentence(question2), u1]

    def e5_intro(self, user_id, app, db_session):
        u1 = "Exercise 5: Pledging to care and support our child"
        u2 = "All mammals are born with an innate capacity to care for children. We're going to take care of our child like we'd take care of ourselves."
        data2 = self.datasets[user_id]
        data = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        return [self.split_sentence(question2), u1, u2]

    def e6_intro(self, user_id, app, db_session):
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
        question3 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question3)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question3)
        return [self.split_sentence(question2), self.split_sentence(question), u1, u2, self.split_sentence(question3)]

    def e7_intro(self, user_id, app, db_session):
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
        return [self.split_sentence(question2), u1, u2]

    def e8_intro(self, user_id, app, db_session):
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
        return [self.split_sentence(question2), u1, u2]

    def e9_intro(self, user_id, app, db_session):
        u1 = "Exercise 9: Overcoming current negative emotions"
        u2 = "If you currently have negative emotions, no worries, you're not alone! I have these little concerns too!"
        mylist = ["I once intended to go to the party but my mom refused to let me hang out. So I did not make it and my friends thought I didn't consider their feelings.",
                  "It has been raining for many days, and I cannot go out to play. This embarasses me a lot.",
                  "There is no air conditioner in my room in this hot summer and that is awful.",
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

        data2 = self.datasets[user_id]
        column2 = data["I want to suggest you this exercise."].dropna()
        question2 = self.get_best_sentence(column2, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question2)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question2)
        return [self.split_sentence(question2), u1, u2, u3]

    def e10_intro(self, user_id, app, db_session):
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
        return [self.split_sentence(question2), u1, u2, self.split_sentence(question)]

    def e11_intro(self, user_id, app, db_session):
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
        return [self.split_sentence(question), u2, u1]

    def e12_intro(self, user_id, app, db_session):
        u1 = "Exercise 12: Laughing on our own"
        u2 = "Have you accomplished something recently?"
        mylist = ["I did chores today.", "I talked to a neighbor today.", "I finished reading a book this month.", "I got good marks for my exams.", "I accomplished to go to gym 5 days each week!",
                  "I performed a presentation of my work. ", "I helped filling out project survey of my friends XD.", "I cooked a meal and my friends really enjoyed it!", "I have finished following a TV series.",
                  "I did research on topics I am interested in!", "I completed two projects in my work!", "I recently have achieved a good average marks for my exam!", "Today when I got out of the gym and went into the chaging room, a person asked me how to lock the cabinet. I told him it needed one pound to lock it and gave him one pound. And then we add each other's contact number."
                  ]
        u3 = np.random.choice(mylist)
        u3 = "For example "+u3
        u4 = "Smile at your accomplishment when you're comfortable, then laugh at it! "
        return [u2, u3, u4, u1]

    def e14_intro(self, user_id, app, db_session):
        u1 = "Exercise 14: Creating your own brand of laughter"
        u2 = "Are you aware that your special laugh can become your brand? That could be implemented into muscle memory, so you subconsciously smile often and can control how much you smile, not feel embarrassed."
        u3 = "The laughter of various characters in One Piece manga series is different. I believe that normal people will not laugh like this, but it does not prevent all kinds of weird laughter from becoming one of the classic symbols of One Piece."
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
        return [u1, u3, u2]

    def e15_intro(self, user_id, app, db_session):
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
        return [self.split_sentence(question2), u1, u2, u3, self.split_sentence(question)]

    def e16_intro(self, user_id, app, db_session):
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
        return [self.split_sentence(question), u2, self.split_sentence(question2), u1]

    def e20_intro(self, user_id, app, db_session):
        u1 = "Exercise 20: Practicing Affirmations"
        u2 = "Are you familiar with Nietzsche or Laozi? Both are famous philosopher. Here's what they said:"
        mylist = ["What does not kill me makes me stronger. — Friedrich Nietzsche's Twilight of the Idols (1888)",
                  "A journey of a thousand miles begins with a single step. —Chapter 64 of the Dao De Jing ascribed to Laozi",
                  "My formula for greatness in a human being is Amor fati: that one wants nothing to be different, not forward, not backward, not in all eternity. Not merely bear what is necessary, still less conceal it—all idealism is mendacity in the face of what is necessary—but love it. — Friedrich Nietzsche, 1888",
                  "To those human beings who are of any concern to me I wish suffering, desolation, sickness, ill-treatment, indignities—I wish that they should not remain unfamiliar with profound self-contempt, the torture of self-mistrust, the wretchedness of the vanquished: I have no pity for them, because I wish them the only thing that can prove today whether one is worth anything or not—that one endures. ― Friedrich Nietzsche, The Will to Power",
                  "Whoever fights monsters should see to it that in the process he does not become a monster. And if you gaze long enough into an abyss, the abyss will gaze back into you.― Friedrich Nietzsche"
                  ]
        u3 = np.random.choice(mylist)
        u4 = "You may find useful to read these affirmations"
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
        return [u2, u3, u4, self.split_sentence(question2), u1]

    def e21_intro(self, user_id, app, db_session):
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
        return [u2, u3, self.split_sentence(question2), u1, self.split_sentence(question)]

    def e22_intro(self, user_id, app, db_session):
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
        return [self.split_sentence(question2), u1, self.split_sentence(question)]

    def e23_intro(self, user_id, app, db_session):
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
        return [u3, u2, self.split_sentence(question),  self.split_sentence(question2), u1]

    def e24_intro(self, user_id, app, db_session):
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
        return [u2, self.split_sentence(question), self.split_sentence(question2), u1]

        "no_more_choice": {
            "model_prompt": lambda user_id, db_session, curr_session, app: self.end_of_suggestions(user_id),
            "choices": {
                "End Session": "ending_prompt",
                "Try Specific Willpower Exercises": "",

            },
            "protocols": {
                "End Session": [],
                "Try Specific Willpower Exercises": []
            },
        },

    def end_of_suggestions(self, user_id, app, db_session):
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
        return self.split_sentence(question)

    if len(self.suggestion_SAT_P) <= 8:
        return ...
    if len(self.suggestion_SAT_P) <= 4:

    def thank_effort(self, user_id, app, db_session):
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

    def cong_effort(self, user_id, app, db_session):
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

    if len(self.suggestion_SAT_P) <= 8:
        return ...
    if len(self.suggestion_SAT_P) <= 4:


if __name__ == "__main__":
    print(get_new_suggestions())

    "get_new_sugg": {
        "model_prompt": lambda user_id, db_session, curr_session, app: self.e1_intro(user_id),
        "choices": {
            "Continue": "ending_prompt",
        },
        "protocols": {
            "Continue": [],
        },
    },


'''

current_protocol_ids



    def end_of_suggestions(self, user_id, app, db_session):
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
        return [self.split_sentence(question), "You could choose to end session or try specific willpower exercises."]

'''
