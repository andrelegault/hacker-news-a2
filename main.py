# -------------------------------------------------------
# Assignment 2
# Written by Andre Parsons-Legault 40031363
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------
import model
import w_filter
import pandas as pd
import matplotlib.pyplot as plt

"""
This is the main script for model generation using a Hacker News dataset (2018, 2019).

13 models are generated in total. 4 models are unique. The last 10 are used to plot performance of various word removal experiments.
The first model is the baseline experiment.
The second is the stopword experiment.
The third is the worthlength experiment.
The fourth is the infrequent word filtering experiment.
"""
if __name__ == '__main__':
    source_filename = input('Please input the name of the file (leave blank to use hns_2018_2019.csv): ')
    if not source_filename:
        source_filename = 'hns_2018_2019.csv'

    data = pd.read_csv(source_filename)

    print('Running baseline experiment [tasks 1, 2]')
    baseline = model.Model('baseline experiment', data)
    baseline.train()
    baseline.test()
    baseline.export_training_model()
    baseline.export_vocabulary('baseline_vocabulary.txt')
    baseline.export_testing_model()
    baseline.export_removed('baseline-remove_word.txt')
    print('Done')

    print('Starting classifier experiments [task 3]')

    print('Running stopword experiment (#1)')
    stopword = model.Model('stopword', data, w_filter.StopWordFilter())
    stopword.train()
    stopword.test()
    stopword.export_vocabulary('stopword-vocabulary.txt')
    stopword.export_training_model('stopword-model.txt')
    stopword.export_testing_model('stopword-result.txt')
    stopword.export_removed('stopword-remove_word.txt')
    print('Done')

    print('Running wordlength experiment (#2)')
    wordlength = model.Model('wordlength', data, w_filter.WordLengthFilter())
    wordlength.train()
    wordlength.test()
    wordlength.export_vocabulary('wordlength-vocabulary.txt')
    wordlength.export_training_model('wordlength-model.txt')
    wordlength.export_testing_model('wordlength-result.txt')
    wordlength.export_removed('wordlength-remove_word.txt')
    print('Done')

    print('Running infrequent word filtering (#3)')
    print('Infrequent word removal with frequency == 1')
    infrequent_1 = model.Model('=1', data)
    infrequent_1.train()
    infrequent_1.remove_where(1)
    infrequent_1.test()
    infrequent_1.export_vocabulary('infrequent_1-vocabulary.txt')
    infrequent_1.export_training_model('infrequent_1-model.txt')
    infrequent_1.export_testing_model('infrequent_1-result.txt')
    infrequent_1.export_removed('infrequent_1-remove_word.txt')
    print('Done')

    print('Infrequent word removal with frequency <= 5')
    infrequent_5 = model.Model('<=5', data)
    infrequent_5.train()
    infrequent_5.remove_where(5)
    infrequent_5.test()
    infrequent_5.export_vocabulary('infrequent_5-vocabulary.txt')
    infrequent_5.export_training_model('infrequent_5-model.txt')
    infrequent_5.export_testing_model('infrequent_5-result.txt')
    infrequent_5.export_removed('infrequent_5-remove_word.txt')
    print('Done')

    print('Infrequent word removal with frequency <= 10')
    infrequent_10 = model.Model('<=10', data)
    infrequent_10.train()
    infrequent_10.remove_where(10)
    infrequent_10.test()
    infrequent_10.export_vocabulary('infrequent_10-vocabulary.txt')
    infrequent_10.export_training_model('infrequent_10-model.txt')
    infrequent_10.export_testing_model('infrequent_10-result.txt')
    infrequent_10.export_removed('infrequent_10-remove_word.txt')
    print('Done')

    print('Infrequent word removal with frequency <= 15')
    infrequent_15 = model.Model('<=15', data)
    infrequent_15.train()
    infrequent_15.remove_where(15)
    infrequent_15.test()
    infrequent_15.export_vocabulary('infrequent_15-vocabulary.txt')
    infrequent_15.export_training_model('infrequent_15-model.txt')
    infrequent_15.export_testing_model('infrequent_15-result.txt')
    infrequent_15.export_removed('infrequent_15-remove_word.txt')
    print('Done')

    print('Infrequent word removal with frequency <= 20')
    infrequent_20 = model.Model('<=20', data)
    infrequent_20.train()
    infrequent_20.remove_where(20)
    infrequent_20.test()
    infrequent_20.export_vocabulary('infrequent_20-vocabulary.txt')
    infrequent_20.export_training_model('infrequent_20-model.txt')
    infrequent_20.export_testing_model('infrequent_20-result.txt')
    infrequent_20.export_removed('infrequent_20-remove_word.txt')
    print('Done')

    print('Top 5% most frequent word removal')
    top_5 = model.Model('<=5%', data)
    top_5.train()
    top_5.remove_top(0.05)
    top_5.test()
    top_5.export_vocabulary('top_5-vocabulary.txt')
    top_5.export_training_model('top_5-model.txt')
    top_5.export_testing_model('top_5-result.txt')
    top_5.export_removed('top_5-remove_word.txt')
    print('Done')

    print('Top 10% most frequent word removal')
    top_10 = model.Model('<=10%', data)
    top_10.train()
    top_10.remove_top(0.10)
    top_10.test()
    top_10.export_vocabulary('top_10-vocabulary.txt')
    top_10.export_training_model('top_10-model.txt')
    top_10.export_testing_model('top_10-result.txt')
    top_10.export_removed('top_10-remove_word.txt')
    print('Done')

    print('Top 15% most frequent word removal')
    top_15 = model.Model('<=15%', data)
    top_15.train()
    top_15.remove_top(0.15)
    top_15.test()
    top_15.export_vocabulary('top_15-vocabulary.txt')
    top_15.export_training_model('top_15-model.txt')
    top_15.export_testing_model('top_15-result.txt')
    top_15.export_removed('top_15-remove_word.txt')
    print('Done')

    print('Top 20% most frequent word removal')
    top_20 = model.Model('<= 20%', data)
    top_20.train()
    top_20.remove_top(0.20)
    top_20.test()
    top_20.export_vocabulary('top_20-vocabulary.txt')
    top_20.export_training_model('top_20-model.txt')
    top_20.export_testing_model('top_20-result.txt')
    top_20.export_removed('top_20-remove_word.txt')
    print('Done')

    print('Top 25% most frequent word removal')
    top_25 = model.Model('<= 25%', data)
    top_25.train()
    top_25.remove_top(0.25)
    top_25.test()
    top_25.export_vocabulary('top_25-vocabulary.txt')
    top_25.export_training_model('top_25-model.txt')
    top_25.export_testing_model('top_25-result.txt')
    top_25.export_removed('top_25-remove_word.txt')
    print('Done')

    fig, (ax1, ax2) = plt.subplots(1, 2)

    plot_1_x = [ 1, 5, 10, 15, 20 ]
    plot_1_y = [ infrequent_1.performance, infrequent_5.performance, infrequent_10.performance, infrequent_15.performance, infrequent_20.performance ]

    plot_2_x = [ 5, 10, 15, 20, 25]
    plot_2_y = [ top_5.performance, top_10.performance, top_15.performance, top_20.performance, top_25.performance ]

    ax1.plot(plot_1_x, plot_1_y)
    ax2.plot(plot_2_x, plot_2_y)

    ax1.set(xlabel='frequency', ylabel='performance')
    ax2.set(xlabel='top %')
    ax1.set_title('Raw frequency count used for removal')
    ax2.set_title('Top % used for removal')
    plt.show()

