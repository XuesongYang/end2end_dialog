''' Description: obtain statistics of data, which is the same to Table 1 in our paper.

    Author      : Xuesong Yang
    Email       : xyang45@illinois.edu
    Created Date: Dec. 31, 2016
'''


import csv
from collections import namedtuple
from prettytable import PrettyTable


def stats(fname):
    Numbers = namedtuple('Numbers', ['utters', 'words', 'tags', 'intents', 'acts'])
    words = set()
    tags = set()
    intents = set()
    acts = set()
    utter_nb = 0
    with open(fname, 'rb') as f:
        for line in csv.DictReader(f):
            utter_nb += 1
            utter = line['User_Utter']
            for word in utter.split():
                words.add(word)
            slot_tags = line['User_SlotTags']
            for tag in slot_tags.split():
                tags.add(tag)
            user_intents = line['User_Intents']
            for intent in user_intents.split(';'):
                intents.add(intent)
            agent_act = line['Agent_Act']
            for act in agent_act.split(';'):
                acts.add(act)

    return Numbers(utters=utter_nb, words=len(words), tags=len(tags), intents=len(intents), acts=len(acts))


def getTable(train, dev, test):
    fieldnames = ['', '#utters', '#words', '#tags', '#intents', '#acts']
    table = PrettyTable()
    table.align = 'r' 
    table.field_names = fieldnames
    table.add_row(['train'] + list(train))
    table.add_row(['dev'] + list(dev))
    table.add_row(['test'] + list(test))
    return table


if __name__ == '__main__':
    train = './data/csv/dstc4.all.w-intent.train.csv'
    dev = './data/csv/dstc4.all.w-intent.dev.csv'
    test = './data/csv/dstc4.all.w-intent.test.csv'
    train_numbers = stats(train)
    dev_numbers = stats(dev)
    test_numbers = stats(test)

    table = getTable(train_numbers, dev_numbers, test_numbers)
    print("Table 1: Statistics of data used in experiments. '#' represents the number of unique items.")
    print(table)
