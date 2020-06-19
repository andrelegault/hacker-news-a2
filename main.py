# -------------------------------------------------------
# Assignment 2
# Written by Andre Parsons-Legault 40031363
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------

import csv
from typing import Dict, Any
from re import findall

import pandas as pd
from math import log

SMOOTH_VALUE = 0.5
LOG_BASE = 10


def training_model_to_file(container, w_categories, filename='model_2018.txt'):
	"""
	Outputs the dataframe to a file named `model_2018.txt` by default. Every row follows the following format:
	index  word  freq_story  cond_story  freq_ask_hn  cond_ask_hn  freq_show_hn  cond_show_hn  freq_poll  cond_poll  <cr>

	freqs is a dict having the following structure:
	{
		'word1' : { 'total': int, 'category_1': int, 'category_2': int, 'category_3': int, 'category_4': int },
		'word2' : { 'total': int, 'category_1': int, 'category_2': int, 'category_3': int, 'category_4': int },
	}

        NOTE: if there are no instances of a post type, the model is not included.
        For instance, the `post` type is missing as the training data does not include a post of this type.
	"""
	words = [x for x in container['words']]
	sorted_words = sorted(words)
	with open(filename, 'w+', encoding='utf-8') as f:
            for j in range(len(sorted_words)):
                sorted_word = sorted_words[j]
                obj_w = container['words'][sorted_word]
                line = "{0}  {1}  ".format(j, sorted_word)
                for w_cat in w_categories:
                    line += "{0}  {1:.6f}  ".format(obj_w['frequencies'][w_cat], obj_w['probabilities'][w_cat])

                line += "\n"

                f.write(line)


def testing_model_to_file(testing_model, categories, filename='baseline-result.txt'):
    with open(filename, 'w+', encoding='utf-8') as f:
        i = 0
        for post_title, obj_post in testing_model.items():
            line = "{0}  {1}  {2}  ".format(i, post_title, obj_post['guess'][0])
            for cat, score in obj_post['probabilities'].items():
                line += "{0}  {1:.6f}  ".format(cat, score)

            validate = "right" if obj_post['answer'] == obj_post['guess'][0] else "wrong"
            line += "{0}  {1}".format(obj_post['answer'], validate)

            line += "\n"

            f.write(line)
            i += 1

def vocabulary_to_file(data, filename='vocabulary.txt') -> None:
    with open(filename, 'w+', encoding='utf-8') as f:
        for word in data['words']:
            f.write(word + "\n")


def get_test_df(data):
    valid_year = data[data.year == 2019]


def filter_year(data, year) -> pd.DataFrame:
    df = data[data.year == year]
    return df


def words_in_title(title):
    return findall(r"([\w'-]+)", title)


def get_categories(df):
    return df['Post Type'].unique()

def nb_word_probability(model_word):
    score = 0

    for probability in model_word['probabilities']:
        score += log(probability, LOG_BASE)

    return score

def determine_categories(testing_model):
    for post_title, obj_post in testing_model.items():
        for category, probability in obj_post['probabilities'].items():
            check = obj_post.get('guess')
            if not check or check[1] < probability:
                testing_model[post_title]['guess'] = (category, probability)

if __name__ == '__main__':
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
    data = pd.read_csv('hns_2018_2019.csv')

    # Get categories for whole data set
    categories = get_categories(data)

    # Training
    training_model: Dict[str, dict] = {'words': {}, 'categories': {}}
    training_data = filter_year(data, 2018)
    training_model['categories'] = { cat: 0 for cat in categories }

    for i, row in training_data.iterrows():
        trimmed_words = words_in_title(row['Title'].lower())
        for trimmed_word in trimmed_words:
            category = row['Post Type']
            if trimmed_word not in training_model['words']:
                training_model['words'][trimmed_word] = {'frequencies': {}, 'probabilities': {}}
                # training_model['words'][l_trimmed_word]['frequencies'] = {'total': 0, 'story': 0, 'poll': 0, 'ask_hn': 0,
                #                                                 'show_hn': 0}
                for cat in categories:
                        training_model['words'][trimmed_word]['frequencies'][cat] = 0
                        training_model['words'][trimmed_word]['probabilities'][cat] = 0
                training_model['words'][trimmed_word]['frequencies']['total'] = 0
            training_model['words'][trimmed_word]['frequencies'][category] += 1
            training_model['words'][trimmed_word]['frequencies']['total'] += 1
            training_model['categories'][category] += 1

    voc_size = len(training_model['words'])
    for word in training_model['words']:
        for cat, count in training_model['categories'].items():
            freq = training_model['words'][word]['frequencies'][cat]
            prob_given_c = (freq + SMOOTH_VALUE) / (count + (voc_size * SMOOTH_VALUE))
            training_model['words'][word]['probabilities'][cat] = prob_given_c

    training_model_to_file(training_model, categories)
    vocabulary_to_file(training_model)

    # Testing
    testing_model: Dict[str, dict] = {}
    testing_data = filter_year(data, 2019)

    cat_scores = {} #  total number of posts of a category divided by total number of posts
    total_posts = len(testing_data)
    for category in categories:
        count_cat = len(testing_data[testing_data['Post Type'] == category])
        cat_scores[category] = count_cat / total_posts

    for i, row in testing_data.iterrows():
        title = row['Title']
        if not testing_model.get(title):
            testing_model[title] = {
                'probabilities': { cat: 0 for cat in categories },
                'answer': row['Post Type']
            }
        trimmed_words = words_in_title(title.lower())
        for category in categories:
            testing_model[title]['probabilities'][category] += cat_scores[category]
            for trimmed_word in trimmed_words:
                training_word = training_model['words'].get(trimmed_word)
                if training_word:
                    word_score = log(training_word['probabilities'][category], LOG_BASE)
                    testing_model[title]['probabilities'][category] += word_score
                    
    determine_categories(testing_model)

    testing_model_to_file(testing_model, categories)
    
