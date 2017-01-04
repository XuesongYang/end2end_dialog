# -*- coding: UTF-8 -*-
''' process the data for system action prediction model that takes as 
    inputs windowed binary vectors of userTags and userIntents, and takes as 
    outputs binary vectors of agentAct
    
    Author      : Xuesong Yang
    Email       : xyang45@illinois.edu
    Created Date: Dec. 31, 2016
'''
from DataSetCSVslotTagging import DataSetCSVslotTagging, vectorizing_binaryVec
from utils import get_windowedVec
import numpy as np


class DataSetCSVagentActPred(DataSetCSVslotTagging):

    def __init__(self, csv_file, window_size=5, train_data=None, flag='train'):
        if flag == 'train':
            self.window_size = window_size
        elif flag == 'test':
            self.window_size = train_data.window_size
        else:
            raise Exception('Unknown flag: {}'.format(flag))
        super(DataSetCSVagentActPred, self).__init__(csv_file, train_data, flag)

    def transform_data(self, maxlen):
        super(DataSetCSVagentActPred, self).transform_data(maxlen)
        tagIntent_vecBin = np.hstack((self.userTag_1hotPad.max(axis=1), self.userIntent_vecBin))
        self.userTagIntent_vecBin = get_windowedVec(tagIntent_vecBin, self.window_size)
        self.agentAct_vecBin, self.agentAct_txt = vectorizing_binaryVec(
            self.agentAct_txt, self.agentAct_vocab_size, self.agentAct2id, prefix='act-')
        assert self.userTagIntent_vecBin.shape[0] == self.agentAct_vecBin.shape[0]
