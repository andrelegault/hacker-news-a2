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


class NBClassifier:
	def __init__(self, data):
		pass  # TODO: set variables

	@staticmethod
	def H(true_occ, false_occ, sum_occ, base=10):
		present = true_occ / sum_occ
		not_present = false_occ / sum_occ

		p1 = present * log(present, base)
		p2 = present * log(not_present, base)

		return p1 + p2

	@staticmethod
	def calc_gain(data):
		pass  # TODO: include formula for gain.


def stats_to_file(container, w_categories, filename='model_2018.txt'):
	"""
	Outputs the dataframe to a file named `model_2018.txt` by default. Every row follows the following format:
	index  word  freq_story  cond_story  freq_ask_hn  cond_ask_hn  freq_show_hn  cond_show_hn  freq_poll  cond_poll  <cr>

	freqs is a dict having the following structure:
	{
		'word1' : { 'total': int, 'story': int, 'ask_hn': int, 'show_hn': int, 'poll': int },
		'word2' : { 'total': int, 'story': int, 'ask_hn': int, 'show_hn': int, 'poll': int },
	}
	"""
	words = [x for x in container['words']]
	sorted_words = sorted(words)
	with open(filename, 'w+', encoding='utf-8') as f:
		for j in range(len(sorted_words)):
			sorted_word = sorted_words[j]
			obj_w = container['words'][sorted_word]
			line = "{0}  {1}  ".format(j, sorted_word)
			for w_cat in w_categories:
				line += "{0}  {1}  ".format(obj_w['frequencies'][w_cat], obj_w['probabilities'][w_cat])

			line += "\n"

			# print(line)

			f.write(line)


def vocabulary_to_file(data, filename='vocabulary.txt') -> None:
	removed_words = data[data['whatever'] in ['banned_words']]


def process_row(row) -> None:
	lowercase = row['Title'].lower()


def get_test_df(data):
	valid_year = data[data.year == 2019]


def get_model_df(data):
	valid_year = data[data.year == 2018]
	return valid_year


def words_in_title(title):
	return findall(r"([\w'-]+)", title)


def get_categories(df):
	return df['Post Type'].unique()


if __name__ == '__main__':
	"""Data has the following structure:
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
	data: Dict[str, dict] = {'words': {}, 'categories': {}}
	df = pd.read_csv('hns_2018_2019.csv')
	categories = get_categories(df)
	print(categories)
	# print(categories)
	for category in categories:
		data['categories'][category] = 0
	year_2018 = get_model_df(df)
	# print(year_2018)

	for i, row in year_2018.iterrows():
		trimmed_words = words_in_title(row['Title'])
		for trimmed_word in trimmed_words:
			l_trimmed_word = trimmed_word.lower()
			category = row['Post Type']
			if l_trimmed_word not in data['words']:
				data['words'][l_trimmed_word] = {'frequencies': {}, 'probabilities': {}}
				# data['words'][l_trimmed_word]['frequencies'] = {'total': 0, 'story': 0, 'poll': 0, 'ask_hn': 0,
				#                                                 'show_hn': 0}
				for cat in categories:
					data['words'][l_trimmed_word]['frequencies'][cat] = 0
					data['words'][l_trimmed_word]['probabilities'][cat] = 0
				data['words'][l_trimmed_word]['frequencies']['total'] = 0
			data['words'][l_trimmed_word]['frequencies'][category] += 1
			data['words'][l_trimmed_word]['frequencies']['total'] += 1
			data['categories'][category] += 1

	voc_size = len(data['words'])
	for word in data['words']:
		for cat, count in data['categories'].items():
			freq = data['words'][word]['frequencies'][cat]
			prob_given_c = (freq + SMOOTH_VALUE) / (count + voc_size)
			data['words'][word]['probabilities'][cat] = prob_given_c

	stats_to_file(data, categories)
