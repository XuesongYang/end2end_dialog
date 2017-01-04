# -*- coding: UTF-8 -*-
''' Data preprocessing for slot tagging and intent prediction.
    Replace the unseen tokens in the test/dev set with <unk> for user intents, user slot tags and agent actions.
    
    Author      : Xuesong Yang
    Email       : xyang45@illinois.edu
    Created Date: Dec. 31, 2016
'''
from DataSetCSV import DataSetCSV
import numpy as np
from keras.preprocessing import sequence
from utils import to_categorical


def vectorizing_zeropad(sentences, maxlen, token2id, prefix=''):
    ''' encode utterance or slot tags into id sequence.
        0s for padding, 1s for unk.
        return a matrix with shape=(sample_nb, maxlen)
        e.g. [[0, 0, 0, 1, 2, 4, 5, ...], [...]]
        Inputs:
            sentences: shape = (sample_nb,), a list of strings
        Outputs:
            zeropad: shape = (sample_nb, maxlen), pre-padded ids
            token_txt: shape = (sample_nb,), new text sequences with unknown words replaced by '<unk>'
    '''
    encode = list()
    token_txt = list()
    for sent in sentences:
        output = list()
        output_txt = list()
        for token in sent.strip().split():
            token = '{}{}'.format(prefix, token.strip())
            if token in token2id:
                output.append(token2id[token])
                output_txt.append(token)
            else:
                # we reserved 1 for <unk>, 0 for <pad>
                output.append(token2id['<{}unk>'.format(prefix)])
                output_txt.append('<{}unk>'.format(prefix))
        encode.append(output)
        token_txt.append(' '.join(output_txt))
    zeropad = sequence.pad_sequences(
        encode, maxlen, padding='pre', truncating='pre')
    return zeropad, np.asarray(token_txt)


def vectorizing_binaryVec(intents, vocab_size, intent2id, prefix=''):
    ''' convert into binary vectors.
        Inputs:
            intents: shape = (sample_nb,)
            vocab_size: scalar, vocabulary size
            intent2id: dict, (token, id) pairs
        Outputs:
            vec: shape = (sample_nb, vocab_size), binary matrix with firing ones when token exists in specific position
            intent_txt: shape = (sample_nb,), a list of text with unknown tokens replaced by '<unk>'
    '''
    vec = np.zeros((intents.shape[0], vocab_size))
    intent_txt = list()
    for intent_idx, intent in enumerate(intents):
        output_txt = set()  # duplicated intent may exist, need to unique
        for token_idx, token in enumerate(intent.strip().split(';')):
            if token == 'null':  # null exists in agent acts, but is not considered as label; while user intent does not include it
                output_txt.add(token)
                continue
            token = '{}{}'.format(prefix, token)
            if token in intent2id:
                vec[intent_idx, intent2id[token] - 1] = 1
                output_txt.add(token)
            else:
                unk = '<{}unk>'.format(prefix)
                vec[intent_idx, intent2id[unk] - 1] = 1
                output_txt.add(unk)
        intent_txt.append(';'.join(sorted(output_txt)))
    return vec, np.asarray(intent_txt)


class DataSetCSVslotTagging(DataSetCSV):

    def __init__(self, csv_file, train_data=None, flag='train'):
        super(DataSetCSVslotTagging, self).__init__(
            csv_file, train_data, flag)

    def getUserUtterMaxlen(self):
        maxlen = 0
        for utter in self.userUtter_txt:
            utter_len = len([x for x in utter.strip().split()])
            if utter_len > maxlen:
                maxlen = utter_len
        return maxlen

    def transform_data(self, maxlen):
        self.maxlen_userUtter = maxlen
        # replace unknown words with <unk> in user utterance, and encode it
        # using word id.
        self.userUtter_encodePad, self.userUtter_txt = vectorizing_zeropad(
            self.userUtter_txt, self.maxlen_userUtter, self.word2id, prefix='')
        # replace unknown tags with <tag-unk> in user slot tags, and encode it
        # as 1hot matrix
        userTag_encodePad, self.userTag_txt = vectorizing_zeropad(
            self.userTag_txt, self.maxlen_userUtter, self.userTag2id, prefix='tag-')
        self.userTag_1hotPad = to_categorical(
            userTag_encodePad, self.userTag_vocab_size)
        # replace unknown intents with <intent-unk> in user intents, and encode
        # it as binary vec
        self.userIntent_vecBin, self.userIntent_txt = vectorizing_binaryVec(
            self.userIntent_txt, self.userIntent_vocab_size, self.userIntent2id, prefix='intent-')


if __name__ == '__main__':
    csv_train = './data/csv/dstc4.all.w-intent.train.csv'
    csv_test = './data/csv/dstc4.all.w-intent.test.csv'
    csv_dev = './data/csv/dstc4.all.w-intent.dev.csv'
    train_data = DataSetCSVslotTagging(csv_train, flag='train')
    dev_data = DataSetCSVslotTagging(
        csv_dev, train_data=train_data, flag='test')
    test_data = DataSetCSVslotTagging(
        csv_test, train_data=train_data, flag='test')
    maxlen_userUtter_train = train_data.getUserUtterMaxlen()
    maxlen_userUtter_dev = dev_data.getUserUtterMaxlen()
    maxlen_userUtter_test = test_data.getUserUtterMaxlen()
    maxlen_userUtter = max(maxlen_userUtter_train,
                           maxlen_userUtter_dev, maxlen_userUtter_test)
    train_data.transform_data(maxlen_userUtter)
    dev_data.transform_data(maxlen_userUtter)
    test_data.transform_data(maxlen_userUtter)
    import ipdb
    ipdb.set_trace()
    print('done')
