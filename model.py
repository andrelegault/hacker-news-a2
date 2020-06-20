import pandas as pd
import w_filter
from re import findall
from math import log
from typing import Dict, Any


SMOOTH_VALUE = 0.5
LOG_BASE = 10


class Model:

    def __init__(self, name, data, word_filter = None):
        self.name = name
        self.data = data
        self.word_filter : Filter = word_filter

        self.training_data = {}
        self.testing_data = {}

        self.training_model: Dict[str, dict] = {'words': {}, 'categories': {}}
        self.testing_model: Dict[str, dict] = {}
        self.categories = self.get_categories()


    def calc_frequencies(self):
        for i, row in self.training_data.iterrows():
            trimmed_words = self.words_in_title(row['Title'].lower())
            for trimmed_word in trimmed_words:
                if not self.word_filter or self.word_filter.is_valid(trimmed_word):
                    category = row['Post Type']
                    if trimmed_word not in self.training_model['words']:
                        self.training_model['words'][trimmed_word] = {'frequencies': {}, 'probabilities': {}}
                        for cat in self.categories:
                                self.training_model['words'][trimmed_word]['frequencies'][cat] = 0
                                self.training_model['words'][trimmed_word]['probabilities'][cat] = 0
                        self.training_model['words'][trimmed_word]['frequencies']['total'] = 0
                    self.training_model['words'][trimmed_word]['frequencies'][category] += 1
                    self.training_model['words'][trimmed_word]['frequencies']['total'] += 1
                    self.training_model['categories'][category] += 1


    def calc_probabilities(self):
        voc_size = len(self.training_model['words'])
        for word in self.training_model['words']:
            for cat, count in self.training_model['categories'].items():
                freq = self.training_model['words'][word]['frequencies'][cat]
                prob_given_c = (freq + SMOOTH_VALUE) / (count + (voc_size * SMOOTH_VALUE))
                self.training_model['words'][word]['probabilities'][cat] = prob_given_c


    def train(self, year=2018):
        """
        Training Data has the following structure:
        {
            'words': {
                string : {
                    'frequencies': { 'total': int, 'story': int, 'poll': int, 'ask_hn': int, 'show_hn': int }
                    'probabilities': { 'story': int, 'poll': int, 'ask_hn': int, 'show_hn': int }
                }
            },
            'categories': { 'poll': int, 'story': int, 'ask_hn': int, 'show_hn': int },
        }
        """
        self.training_data = self.filter_year(year)
        self.training_model['categories'] = { cat: 0 for cat in self.categories }

        self.calc_frequencies()

        self.calc_probabilities()


    def calc_scores(self):
        cat_scores = {} #  total number of posts of a category divided by total number of posts
        total_posts = len(self.testing_data)
        for category in self.categories:
            count_cat = len(self.testing_data[self.testing_data['Post Type'] == category])
            cat_scores[category] = count_cat / total_posts

        for i, row in self.testing_data.iterrows():
            title = row['Title']
            if not self.testing_model.get(title):
                self.testing_model[title] = {
                    'probabilities': { cat: 0 for cat in self.categories },
                    'answer': row['Post Type']
                }
            trimmed_words = self.words_in_title(title.lower())
            for category in self.categories:
                self.testing_model[title]['probabilities'][category] += cat_scores[category]
                for trimmed_word in trimmed_words:
                    if not self.word_filter or self.word_filter.is_valid(trimmed_word):
                        training_word = self.training_model['words'].get(trimmed_word)
                        if training_word:
                            word_score = log(training_word['probabilities'][category], LOG_BASE)
                            self.testing_model[title]['probabilities'][category] += word_score

    def test(self, year=2019):
        """Testing."""
        self.testing_data = self.filter_year(year)
        self.calc_scores()
        self.calc_highest_cat()
        self.calc_performance()

    
    def get_categories(self):
        """Get categories for whole data set."""
        return self.data['Post Type'].unique()

    def filter_year(self, year):
        return self.data[self.data.year == year]


    def words_in_title(self, title):
        return findall(r"([\w'-]+)", title)


    def calc_highest_cat(self):
        for post_title, obj_post in self.testing_model.items():
            for category, probability in obj_post['probabilities'].items():
                check = obj_post.get('guess')
                if not check or check[1] < probability:
                    self.testing_model[post_title]['guess'] = (category, probability)


    def export_vocabulary(self, filename='vocabulary.txt') -> None:
        with open('output/' + filename, 'w+', encoding='utf-8') as f:
            for word in self.training_model['words']:
                f.write(word + "\n")


    def export_training_model(self, filename='model_2018.txt'):
        """
        Outputs the dataframe to a file named `model_2018.txt` by default. Every row follows the following format:
        index  word  freq_story  cond_story  freq_ask_hn  cond_ask_hn  freq_show_hn  cond_show_hn  freq_poll  cond_poll  <cr>

        freqs is a dict having the following structure:
        {
                'word1' : { 'total': int, 'category_1': int, 'category_2': int, 'category_3': int, 'category_4': int },
                'word2' : { 'total': int, 'category_1': int, 'category_2': int, 'category_3': int, 'category_4': int },
        }

        NOTE: if there are no instances of a post type, the no output rows will have it.
        For instance, the `post` type is missing as the training data does not include a post of this type.
        """
        words = [x for x in self.training_model['words']]
        sorted_words = sorted(words)
        with open('output/' + filename, 'w+', encoding='utf-8') as f:
            for j in range(len(sorted_words)):
                sorted_word = sorted_words[j]
                obj_w = self.training_model['words'][sorted_word]
                line = "{0}  {1}  ".format(j, sorted_word)
                for w_cat in self.categories:
                    line += "{0}  {1:.6f}  ".format(obj_w['frequencies'][w_cat], obj_w['probabilities'][w_cat])

                line += "\n"

                f.write(line)


    def export_testing_model(self, filename='baseline-result.txt'):
        with open('output/' + filename, 'w+', encoding='utf-8') as f:
            i = 0
            for post_title, obj_post in self.testing_model.items():
                line = "{0}  {1}  {2}  ".format(i, post_title, obj_post['guess'][0])
                for cat, score in obj_post['probabilities'].items():
                    line += "{0}  {1:.6f}  ".format(cat, score)

                validate = "right" if obj_post['answer'] == obj_post['guess'][0] else "wrong"
                line += "{0}  {1}".format(obj_post['answer'], validate)

                line += "\n"

                f.write(line)
                i += 1

    def remove_where(self, bound):
        if bound == 1:
            bad_words = { key for key, obj_wo in self.training_model['words'].items() if obj_wo['frequencies']['total'] == 1 }
        else:
            bad_words = { key for key, obj_wo in self.training_model['words'].items() if obj_wo['frequencies']['total'] <= bound }
        for bad_word in bad_words:
            del self.training_model['words'][bad_word]

    def remove_top(self, percentage):
        i_bound = round(len(self.training_model['words']) * percentage)
        by_total_freq = sorted(self.training_model['words'].items(), reverse=True, key=lambda word: word[1]['frequencies']['total'])
        words_removed = by_total_freq[:i_bound-1]
        for to_be_removed in words_removed:
            del self.training_model['words'][to_be_removed[0]]

    def calc_performance(self):
        A, B = 0, 0
        for post in self.testing_model.values():
            if post['answer'] == post['guess'][0]:
                A += 1
            else:
                B += 1

        self.performance = (A / (A + B))

