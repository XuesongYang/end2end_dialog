# -*- coding: UTF-8 -*-
''' process the data for end-to-end joint model that takes as 
    inputs consecutive window_sized user utterances, and takes as
    outputs slot tagging, user intents, and system actions.
    
    Author      : Xuesong Yang
    Email       : xyang45@illinois.edu
    Created Date: Dec. 31, 2016
'''
from DataSetCSVslotTagging import DataSetCSVslotTagging
from DataSetCSVslotTagging import vectorizing_binaryVec
from utils import get_windowedVec


class DataSetCSVjoint(DataSetCSVslotTagging):

    def __init__(self, csv_file, window_size=5, train_data=None, flag='train'):
        if flag == 'train':
            self.window_size = window_size
        elif flag == 'test':
            self.window_size = train_data.window_size
        else:
            raise Exception('Unknown flag: {}'.format(flag))
        super(DataSetCSVjoint, self).__init__(csv_file, train_data, flag)

    def transform_data(self, maxlen_userUtter):
        super(DataSetCSVjoint, self).transform_data(maxlen_userUtter)
        # process windowed user utter
        self.userUtter_encodePad_window = get_windowedVec(self.userUtter_encodePad, self.window_size)
        # process windowed user slot tags
        self.userTag_1hotPad_window = get_windowedVec(self.userTag_1hotPad, self.window_size)
        # process windowed user intents
        self.userIntent_vecBin_window = get_windowedVec(self.userIntent_vecBin, self.window_size)
        # process agent actions
        self.agentAct_vecBin, self.agentAct_txt = vectorizing_binaryVec(
            self.agentAct_txt, self.agentAct_vocab_size, self.agentAct2id, prefix='act-')
