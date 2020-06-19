# -------------------------------------------------------
# Assignment 2
# Written by Andre Parsons-Legault 40031363
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------
import pandas as pd
import model


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

    task_1 = model.Model('Task 1', data)

    task_1.train()
    task_1.test()

    task_1.export_training_model()
    task_1.export_vocabulary()
    task_1.export_testing_model()
    
