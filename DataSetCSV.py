# -*- coding: UTF-8 -*-
''' Load .csv file with the following field names:
        Episode_ID, User_Utter, User_SlotTags, User_Intents, Agent_PrevAct, Agent_Utter, Agent_Act
        e.g. 0,"BOS hi , good- EOS",O O O O O,FOL_OPENING,BOS my name is mathew . EOS,O O O O O O O,INI_OPENING,FOL_INFO;FOL_OPENING

    and then transform it into an instance of this class. Each user utterance may correspond to an intent or system act with several duplicated components delimited by \';\'. This class only consider the unique components for intents and system actions.  
    
    Author      : Xuesong Yang
    Email       : xyang45@illinois.edu
    Created Date: Dec. 31, 2016
'''
from utils import checkExistence
import csv
from collections import OrderedDict
import numpy as np


def getOrderedDict(samples, prefix='intent-', delimiter=';'):
    """ get token2id, id2token, vocab_size for intent, act, tag, or word.
        need to reserve 1 for '<unk>', and reserve 0 for '<pad>'. 
        note: there could be duplicated tokens in a single sample, e.g. a;a;b;b;c
        token_vocab_size should be decreased by 1 since <pad> should be excluded.
        assume each sample contains a single utterance.
    """
    odList = OrderedDict([('<pad>', 0), ('<{}unk>'.format(prefix), 1)])
    vocab_size = len(odList.keys())
    for sent in samples:
        for token in sent.strip().split(delimiter):
            if token == 'null':  # null exists in agent acts, but is not considered as label
                continue
            token = '{}{}'.format(prefix, token.strip())
            if token not in odList:
                odList[token] = vocab_size
                vocab_size += 1
    assert vocab_size == len(odList.keys()), 'Wrong number of vocab size.'
    rodList = OrderedDict([(val, key) for (key, val) in odList.iteritems()])
    return (odList, rodList, vocab_size - 1)


class DataSetCSV(object):

    def __init__(self, csv_file, train_data=None, flag='train'):
        self.csv_file = csv_file
        checkExistence(self.csv_file)
        self._load_data()
        if flag == 'test':
            assert isinstance(
                train_data, DataSetCSV), 'train_data is not an instance of DataSetCSV'
            self.userIntent2id = train_data.userIntent2id
            self.id2userIntent = train_data.id2userIntent
            self.userIntent_vocab_size = train_data.userIntent_vocab_size
            self.userTag2id = train_data.userTag2id
            self.id2userTag = train_data.id2userTag
            self.userTag_vocab_size = train_data.userTag_vocab_size
            self.agentAct2id = train_data.agentAct2id
            self.id2agentAct = train_data.id2agentAct
            self.agentAct_vocab_size = train_data.agentAct_vocab_size
            self.word2id = train_data.word2id
            self.id2word = train_data.id2word
            self.word_vocab_size = train_data.word_vocab_size
            self.userTagIntent2id = train_data.userTagIntent2id
            self.id2userTagIntent = train_data.id2userTagIntent
            self.userTagIntent_vocab_size = train_data.userTagIntent_vocab_size
            self.userTagIntentAgentPrevAct2id = train_data.userTagIntentAgentPrevAct2id
            self.id2userTagIntentAgentPrevAct = train_data.id2userTagIntentAgentPrevAct
            self.userTagIntentAgentPrevAct_vocab_size = train_data.userTagIntentAgentPrevAct_vocab_size
        elif flag == 'train':
            self._get_params()
        else:
            raise Exception('Unknown flag found: {}'.format(flag))

    def _load_data(self):
        self.userUtter_txt = list()
        self.userTag_txt = list()
        self.userIntent_txt = list()
        self.agentUtter_txt = list()
        self.agentPrevAct_txt = list()
        self.agentAct_txt = list()
        with open(self.csv_file, 'rb') as fcsv:
            for line_dct in csv.DictReader(fcsv):
                user_utter = line_dct['User_Utter']
                user_intent = line_dct['User_Intents']
                user_slotTags = line_dct['User_SlotTags']
                agent_utter = line_dct['Agent_Utter']
                agent_prevAct = line_dct['Agent_PrevAct']
                agent_act = line_dct['Agent_Act']
                self.userUtter_txt.append(user_utter)
                self.userIntent_txt.append(user_intent)
                self.userTag_txt.append(user_slotTags)
                self.agentUtter_txt.append(agent_utter)
                self.agentPrevAct_txt.append(agent_prevAct)
                self.agentAct_txt.append(agent_act)
        self.userUtter_txt = np.asarray(self.userUtter_txt)
        self.userTag_txt = np.asarray(self.userTag_txt)
        self.userIntent_txt = np.asarray(self.userIntent_txt)
        self.agentUtter_txt = np.asarray(self.agentUtter_txt)
        self.agentPrevAct_txt = np.asarray(self.agentPrevAct_txt)
        self.agentAct_txt = np.asarray(self.agentAct_txt)

    def _get_params(self):
        '''
            [Q] how to deal with the case: the tag, intent, act in test or dev set do not
                exist in train set, but they are predefined before labeling the data?
            [A] the label space is only constrained on train data, and therefore <unk> is used for others. 
            all token2id and id2token include (<pad>, 0), (<unk>, 1).
            all token_vocab_size does not take into account <pad>.
        '''
        (self.userIntent2id, self.id2userIntent, self.userIntent_vocab_size) = getOrderedDict(
            self.userIntent_txt, prefix='intent-', delimiter=';')
        (self.agentAct2id, self.id2agentAct, self.agentAct_vocab_size) = getOrderedDict(
            self.agentAct_txt, prefix='act-', delimiter=';')
        (self.userTag2id, self.id2userTag, self.userTag_vocab_size) = getOrderedDict(
            self.userTag_txt, prefix='tag-', delimiter=None)
        (self.word2id, self.id2word, self.word_vocab_size) = getOrderedDict(
            self.userUtter_txt, prefix='', delimiter=None)
        self.userTagIntent2id = OrderedDict()
        self.userTagIntentAgentPrevAct2id = OrderedDict()
        for (tag, tag_id) in self.userTag2id.iteritems():  # <pad> is included
            self.userTagIntentAgentPrevAct2id.update({tag: tag_id})
            self.userTagIntent2id.update({tag: tag_id})
        self.userTagIntentAgentPrevAct_vocab_size = len(
            self.userTagIntentAgentPrevAct2id.keys())
        self.userTagIntent_vocab_size = len(
            self.userTagIntent2id.keys())
        for intent in self.userIntent2id.keys()[1:]:
            self.userTagIntentAgentPrevAct2id.update(
                {intent: self.userTagIntentAgentPrevAct_vocab_size})
            self.userTagIntentAgentPrevAct_vocab_size += 1
            self.userTagIntent2id.update(
                {intent: self.userTagIntent_vocab_size})
            self.userTagIntent_vocab_size += 1
        self.userTagIntent_vocab_size -= 1  # exclude <pad>
        self.id2userTagIntent = OrderedDict(
            [(val, key) for (key, val) in self.userIntent2id.iteritems()])
        for prevAct in self.agentAct2id.keys()[1:]:
            self.userTagIntentAgentPrevAct2id.update({
                prevAct: self.userTagIntentAgentPrevAct_vocab_size})
            self.userTagIntentAgentPrevAct_vocab_size += 1
        self.userTagIntentAgentPrevAct_vocab_size -= 1
        self.id2userTagIntentAgentPrevAct = OrderedDict(
            [(val, key) for (key, val) in self.userTagIntentAgentPrevAct2id.iteritems()])


if __name__ == '__main__':
    csv_train = './data/csv/dstc4.all.w-intent.train.csv'
    csv_dev = './data/csv/dstc4.all.w-intent.dev.csv'
    train_data = DataSetCSV(csv_train, flag='train')
    dev_data = DataSetCSV(csv_dev, train_data=train_data, flag='test')
    import ipdb; ipdb.set_trace()
    print 'done'
