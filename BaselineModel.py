# -*- coding: UTF-8 -*-
''' Baseline system using CRFtagger and SVM to perform NLU and SAP, respectively.

    Training Process: training two models separately.
    Test Process: raw text --> CRFtagger with lexical features --> user tag sequence
                  --> reshape into binary vecor --> OneVsRestClassifier(LinearSVC)
                  --> evaluate using precision-recall curve

    Author      : Xuesong Yang
    Email       : xyang45@illinois.edu
    Created Date: Dec. 31, 2016
'''
import argparse
from utils import eval_intentPredict, eval_actPred, getActPred, writeTxt, checkExistence 
from DataSetCSVagentActPred import DataSetCSVagentActPred
import numpy as np
from nltk.tag import CRFTagger
import os
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support
import glob


def getUtterList(sents):
    utter_lst = list()
    for sent in sents:
        words = [word for word, tag in sent]
        utter_lst.append(words)
    return utter_lst


def getTagBinaryVector(userTags_pred, userTag2id, userTag_vocab_size):
    ''' userTags_pred: [[(w1, tag1), (w2, tag2)],[]]
    '''
    vec = np.zeros((len(userTags_pred), userTag_vocab_size))
    for idx, sample in enumerate(userTags_pred):
        for word, tag in sample:
            vec[idx, int(userTag2id[tag]) - 1] = 1
    return vec


def writeUtterTag(sents, fname):
    ''' write user utter and user tags into a file. 
        each line constrains with the format: w1 w2 w3\ttag1 tag2 tag3
        Inputs:
            sents: a list of lists, [[(u'w1', tag1), (u'w2', tag2), (u'w3', tag3)], [...], [...]]
            fname: target file name
    '''
    with open(fname, 'wb') as f:
        for sent in sents:
            tmp_words = list()
            tmp_tags = list()
            for word, tag in sent:
                tmp_words.append(word.encode('utf-8'))
                tmp_tags.append(tag.replace('tag-', '', 1))
            sent_str = '{}\t{}'.format(' '.join(tmp_words), ' '.join(tmp_tags))
            f.write('{}\n'.format(sent_str))


def eval_tagPredBaseline(y_true, y_pred, userTag2id, tag_vocab_size):
    ''' calculate performance score
        Input:
            y_true: true tags, [[(u'w1', tag1), (u'w2', tag2), (u'w3', tag3)], [...], [...]]
            y_pred: predicted tags, [[(u'w1', tag1), (u'w2', tag2), (u'w3', tag3)], [...], [...]]
            userTag2id: dict, mapping between id and tag
        Output:
            precision, recall, f1score, accuracy_frame.
    '''
    # calculate token-level scores
    assert len(y_true) == len(y_pred), 'sample_nb is not the same.'
    true_tag_1hot_noO = list()
    for sample_true in y_true:
        tmp_array = np.zeros((len(sample_true), tag_vocab_size))
        for idx, (_, tag_true) in enumerate(sample_true):
            if tag_true != 'tag-O':
                tmp_array[idx, userTag2id[tag_true] - 1] = 1
        true_tag_1hot_noO.extend(tmp_array.tolist())
    pred_tag_1hot_noO = list()
    for sample_pred in y_pred:
        tmp_array = np.zeros((len(sample_pred), tag_vocab_size))
        for idx, (_, tag_pred) in enumerate(sample_pred):
            if tag_pred != 'tag-O':
                tmp_array[idx, userTag2id[tag_pred] - 1] = 1
        pred_tag_1hot_noO.extend(tmp_array.tolist())
    true_tag_1hot_noO = np.asarray(true_tag_1hot_noO)
    pred_tag_1hot_noO = np.asarray(pred_tag_1hot_noO)
    assert true_tag_1hot_noO.shape == pred_tag_1hot_noO.shape, 'shape is not the same.'
    precision, recall, fscore, _ = precision_recall_fscore_support(true_tag_1hot_noO.ravel(), pred_tag_1hot_noO.ravel(), beta=1.0, pos_label=1, average='binary')
    # calculate frame-level scores
    hit = 0.
    sample_nb = len(y_true)
    for sample_true, sample_pred in zip(y_true, y_pred):
        str_true = ' '.join([x_true for (w, x_true) in sample_true])
        str_pred = ' '.join([x_pred for (w, x_pred) in sample_pred])
        if str_true == str_pred:
            hit += 1.
    accuracy_frame = hit * 1. / sample_nb
    return (precision, recall, fscore, accuracy_frame)


def trainSlotTaggingModel(**argparams):
    ''' train slot tagging model using human annotated data
        Input: userUtter
        Output: target userTags
    '''
    print('<Slot Tagging Model>')
    slotTagging_model = SlotTaggingModel(**argparams)
    slotTagging_model.train(verbose=False)
    return slotTagging_model


def trainIntentModel(train_data, dev_data, model_folder):
    ''' train intent prediction model using human annotated data
        Input: bag-of-words of user utterances, Output: indicator matrix of agent actions
    '''
    print('<Intent Prediction Model>')
    train_X_bow = getBagOfWords(train_data.userUtter_encodePad, train_data.word_vocab_size)
    dev_X_bow = getBagOfWords(dev_data.userUtter_encodePad, dev_data.word_vocab_size)
    intent_kwargs = {'train_X': train_X_bow,
                     'train_y_vecBin': train_data.userIntent_vecBin,
                     'dev_X': dev_X_bow,
                     'dev_y_vecBin': dev_data.userIntent_vecBin,
                     'dev_utter_txt': dev_data.userUtter_txt,
                     'dev_y_txt': dev_data.userIntent_txt,
                     'id2token': train_data.id2userIntent,
                     'prefix': 'intent-',
                     'task_name': 'pipeline',
                     'model_folder': model_folder}
    intent_model = MultiLabelClassifier(**intent_kwargs)
    intent_model.train(verbose=False)
    return intent_model


def trainActModel(train_data, dev_data, model_folder):
    print('<System Action Prediction Model>')
    sap_kwargs = {'train_X': train_data.userTagIntent_vecBin[:, -1],
                  'train_y_vecBin': train_data.agentAct_vecBin,
                  'dev_X': dev_data.userTagIntent_vecBin[:, -1],
                  'dev_y_vecBin': dev_data.agentAct_vecBin,
                  'dev_utter_txt': dev_data.userUtter_txt,
                  'dev_y_txt': dev_data.agentAct_txt,
                  'id2token': train_data.id2agentAct,
                  'prefix': 'act-',
                  'task_name': 'oracle',
                  'model_folder': model_folder}
    sap_model = MultiLabelClassifier(**sap_kwargs)
    sap_model.train(verbose=False)
    return sap_model


def getBagOfWords(utter_encodePad, word_vocab_size):
    ''' calculate BoW feature 
        Input: an 2-darray user utterance with zero padding
        Output: 2-darray
    '''
    bow = np.zeros((utter_encodePad.shape[0], word_vocab_size))
    for sample_idx, sample in enumerate(utter_encodePad):
        bow[sample_idx] = np.bincount(sample, minlength=word_vocab_size + 1)[1:]
    return bow


class SlotTaggingModel(object):

    def __init__(self, **argparams):
        self.train_data = argparams['train_data']
        if self.train_data is not None:
            assert isinstance(self.train_data, DataSetCSVagentActPred)
        self.model_folder = argparams['model_folder']
        self.model_fname = '{}/slotTagging.model'.format(self.model_folder)

    def train(self, verbose=True):
        assert self.train_data is not None, 'train_data is required.'
        print('\ttraining ...')
        # transform data
        instance_list = self._transform_data(self.train_data)
        userUtterTag_train_fname = '{}/userUtterTag_train.txt'.format(self.model_folder)
        writeUtterTag(instance_list, userUtterTag_train_fname)
        print('\ttrain_data={}'.format(userUtterTag_train_fname))
        # train model
        self.model = CRFTagger(verbose=verbose)
        self.model.train(instance_list, self.model_fname)
        print('\tmodel_fname={}'.format(self.model_fname))
        print('\tsaving model ...')

    def _transform_data(self, data):
        ''' convert textual utter and user tags into a list of lists that contain lists of (w, t) pairs
        '''
        userUtter_txt = data.userUtter_txt
        userTag_txt = data.userTag_txt
        instance_list = list()
        for words, tags in zip(userUtter_txt, userTag_txt):
            instance = [(word.strip(), tag.strip()) for word, tag in zip(words.decode('utf-8').strip().split(), tags.decode('utf-8').strip().split())]
            instance_list.append(instance)
        return instance_list

    def predict(self, test_data):
        '''return a list of lists, [[(w1, tag1), (w2, tag2), (w3, tag3)], [...], [...]]
        '''
        assert test_data is not None, 'test_data is required.'
        assert isinstance(test_data, DataSetCSVagentActPred)
        print('\tpredicting Slot Tags ...')
        # transform data
        instance_list = self._transform_data(test_data)
        userUtterTag_test_fname = '{}/userUtterTag_test.target'.format(self.model_folder)
        writeUtterTag(instance_list, userUtterTag_test_fname)
        print('\ttag_target={}'.format(userUtterTag_test_fname))
        instance_utter_list = getUtterList(instance_list)
        # testing
        results = self.model.tag_sents(instance_utter_list)
        self.result_fname = '{}/userUtterTag_test.pred'.format(self.model_folder)
        print('\ttag_pred={}'.format(self.result_fname))
        writeUtterTag(results, self.result_fname)
        precision, recall, fscore, accuracy_frame = eval_tagPredBaseline(instance_list, results, test_data.userTag2id, test_data.userTag_vocab_size)
        print('\tprecision={:.4f}, recall={:.4f}, fscore={:.4f}, accuracy_frame={:.4f}'.format(precision, recall, fscore, accuracy_frame))
        return results

    def load_model(self, verbose=True):
        print('\tloading model ...')
        self.model = CRFTagger(verbose=verbose)
        self.model.set_model_file(self.model_fname)


class MultiLabelClassifier(object):
    ''' OneVsRestClassifier(LinearSVC) that is suitable for either
        multi-label intent prediction or system action prediction
        Input: binary vector, output: multi-label probs
    '''
    def __init__(self, **argparams):
        self.train_X = argparams['train_X']
        self.train_y_vecBin = argparams['train_y_vecBin']
        self.dev_X = argparams['dev_X']
        self.dev_y_vecBin = argparams['dev_y_vecBin']
        self.dev_utter_txt = argparams['dev_utter_txt']
        self.dev_y_txt = argparams['dev_y_txt']
        self.model_folder = argparams['model_folder']
        self.prefix = argparams['prefix']
        self.task_name = argparams['task_name']  # 'oracle' or 'pipeline'
        self.id2token = argparams['id2token']

    def train(self, verbose=True):
        assert self.train_X is not None and self.train_y_vecBin is not None, 'train_X and train_y_vecBin are required.'
        assert self.dev_X is not None and self.dev_y_vecBin is not None, 'dev_X and dev_y_vecBin are required.'
        print('\ttraining ...')
        self.model = OneVsRestClassifier(SVC(kernel='linear', probability=True, verbose=verbose))
        self.model.fit(self.train_X, self.train_y_vecBin)
        probs = self.model.predict_proba(self.dev_X)
        # evaluation for user intent
        precision, recall, fscore, accuracy_frame, self.threshold = eval_intentPredict(probs, self.dev_y_vecBin)
        print('\teval_dev: precision={:.4f}, recall={:.4f}, fscore={:.4f}, accuracy_frame={:.4f}, threshold={:.4f}'.format(precision, recall, fscore, accuracy_frame, self.threshold))
        # write prediction results
        dev_txt = getActPred(probs, self.threshold, self.id2token)
        dev_pred_fname = '{}/{}_{}dev.pred'.format(self.model_folder, self.task_name, self.prefix)
        writeTxt(dev_txt, dev_pred_fname, prefix=self.prefix, delimiter=';')
        print('\tdev_pred={}'.format(dev_pred_fname))
        # write target dev
        dev_target_fname = '{}/{}_{}dev.target'.format(self.model_folder, self.task_name, self.prefix)
        writeTxt(self.dev_y_txt, dev_target_fname, prefix=self.prefix, delimiter=';')
        print('\tdev_target={}'.format(dev_target_fname))
        # write utter dev
        dev_utter_fname = '{}/utter_dev.txt'.format(self.model_folder) 
        writeTxt(self.dev_utter_txt, dev_utter_fname, prefix='', delimiter=None)
        print('\tdev_utter={}'.format(dev_utter_fname))
        # save model
        self.model_fname = '{}/{}_{}model_F1={:.4f}_FrameAcc={:.4f}_th={:.4f}.npz'.format(
            self.model_folder, self.task_name, self.prefix, fscore, accuracy_frame, self.threshold)
        np.savez_compressed(self.model_fname, model=self.model, threshold=self.threshold)
        print('\tsaving model: {}'.format(self.model_fname))

    def predict(self, X, y_vecBin, X_utter_txt, y_txt):
        print('\tpredicting ...')
        probs = self.model.predict_proba(X)
        preds_indicator, precision, recall, fscore, accuracy_frame = eval_actPred(probs, y_vecBin, self.threshold)
        print('\tprecision={:.4f}, recall={:.4f}, fscore={:.4f}, accuracy_frame={:.4f}'.format(precision, recall, fscore, accuracy_frame))
        # write prediction test results
        pred_txt = getActPred(probs, self.threshold, self.id2token)
        pred_fname = '{}/{}_{}test.pred'.format(self.model_folder, self.task_name, self.prefix)
        writeTxt(pred_txt, pred_fname, prefix=self.prefix, delimiter=';')
        print('\ttest_pred={}'.format(pred_fname))
        # write target test 
        target_fname = '{}/{}_{}test.target'.format(self.model_folder, self.task_name, self.prefix)
        writeTxt(y_txt, target_fname, prefix=self.prefix, delimiter=';')
        print('\ttest_target={}'.format(target_fname))
        # write utter test 
        utter_fname = '{}/utter_test.txt'.format(self.model_folder) 
        writeTxt(X_utter_txt, utter_fname, prefix='', delimiter=None)
        print('\ttest_utter={}'.format(utter_fname))
        return preds_indicator

    def load_model(self, model_fname):
        assert os.path.exists(model_fname), 'model_fname is required.'
        print('\tloading model: {}'.format(model_fname))
        self.model_fname = model_fname
        npz_fname = np.load(self.model_fname)
        self.model = npz_fname['model'][()]
        self.threshold = np.float(npz_fname['threshold'][()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-npz', dest='data_npz', help='.npz file that is saved as an instance of DataSetCSVagentActPred class, including train, dev, and test data, repectively.')
    parser.add_argument('--train', dest='train_only', action='store_true',
                        help='perform training procedures for CRFtagger and OneVsRest SVMs if this option is activated.')
    parser.add_argument('--test', dest='test_only', action='store_true',
                        help='perform testing for oracle models (CRFtagger, OneVsRest SVMs) and their pipelined model if this option is activated.')
    parser.add_argument('--model-folder', dest='model_folder', help='model folder')
    args = parser.parse_args()
    argparams = vars(args)
    train_only = argparams['train_only']
    test_only = argparams['test_only']
    assert train_only or test_only, 'Argument required: either --train, --test, or both.'

    # load train and test data
    npz_file = argparams['data_npz']
    checkExistence(npz_file)
    data_npz = np.load(npz_file)
    train_data = data_npz['train_data'][()]
    dev_data = data_npz['dev_data'][()]
    test_data = data_npz['test_data'][()]

    ###################################################################################
    ##### Training SlotTagging, Intent Prediction, and AgentAct Prediction models #####
    ###################################################################################
    if train_only:
        if argparams['model_folder'] is None:
            pid = os.getpid()
            argparams['model_folder'] = './model/baseline_{}'.format(pid)
        if not os.path.exists(argparams['model_folder']):
            os.makedirs(argparams['model_folder'])

        # slot tagging model
        argparams['train_data'] = train_data
        slotTagging_model = trainSlotTaggingModel(**argparams)

        # user intent prediction model
        userIntent_model = trainIntentModel(train_data, dev_data, argparams['model_folder'])

        # agent action prediction model
        agentAct_model = trainActModel(train_data, dev_data, argparams['model_folder'])

    ###################################################################################
    ##### Testing SlotTagging, Intent Prediction, and AgentAct Prediction models #####
    ###################################################################################
    if test_only:
        assert os.path.exists(argparams['model_folder']), 'model_folder is required.' 
        # Oracle results of agent action prediction
        print('<Oracle Results of Agent Action Prediction>')
        oracle_task_name = 'oracle'
        oracle_prefix = 'act-'
        sap_model_fname = glob.glob('{}/{}_{}*.npz'.format(argparams['model_folder'], oracle_task_name, oracle_prefix))[0]
        sap_kwargs = {'train_X': None,
                      'train_y_vecBin': None,
                      'dev_X': None,
                      'dev_y_vecBin': None,
                      'dev_utter_txt': None,
                      'dev_y_txt': None,
                      'id2token': train_data.id2agentAct,
                      'prefix': oracle_prefix,
                      'task_name': oracle_task_name,
                      'model_folder': argparams['model_folder']}
        sap_model = MultiLabelClassifier(**sap_kwargs)
        sap_model.load_model(sap_model_fname)
        _ = sap_model.predict(test_data.userTagIntent_vecBin[:, -1], test_data.agentAct_vecBin, test_data.userUtter_txt, test_data.agentAct_txt)

        # Pipelined results of slot tagging
        print('<Pipelined Results of Slot Tagging>')
        argparams['train_data'] = None
        userTag_model = SlotTaggingModel(**argparams)
        userTag_model.load_model()
        userTag_pred = userTag_model.predict(test_data)  # [[(w1, tag1), (w2, tag2)], ... ]
        userTag_vecBin_pred = getTagBinaryVector(userTag_pred, test_data.userTag2id, test_data.userTag_vocab_size)

        # Pipelined results of intent prediction
        print('<Pipelined Results of User Intent Prediction>')
        userIntent_task_name = 'pipeline'
        userIntent_prefix = 'intent-'
        userIntent_model_fname = glob.glob('{}/{}_{}*.npz'.format(argparams['model_folder'], userIntent_task_name, userIntent_prefix))[0]
        userIntent_kwargs = {'train_X': None,
                             'train_y_vecBin': None,
                             'dev_X': None,
                             'dev_y_vecBin': None,
                             'dev_utter_txt': None,
                             'dev_y_txt': None,
                             'id2token': train_data.id2userIntent,
                             'prefix': userIntent_prefix,
                             'task_name': userIntent_task_name,
                             'model_folder': argparams['model_folder']}
        userIntent_model = MultiLabelClassifier(**userIntent_kwargs)
        userIntent_model.load_model(userIntent_model_fname)
        userUtter_X_bow = getBagOfWords(test_data.userUtter_encodePad, test_data.word_vocab_size)
        userIntent_pred_indicator = userIntent_model.predict(userUtter_X_bow, test_data.userIntent_vecBin, test_data.userUtter_txt, test_data.userIntent_txt)

        # Pipelined results of agent action prediction
        print('<Pipelined Results of Agent Action Prediction>')
        act_task_name = 'pipeline'
        act_prefix = 'act-'
        act_model_fname = glob.glob('{}/{}_{}*.npz'.format(argparams['model_folder'], 'oracle', act_prefix))[0]
        act_kwargs = {'train_X': None,
                      'train_y_vecBin': None,
                      'dev_X': None,
                      'dev_y_vecBin': None,
                      'dev_utter_txt': None,
                      'dev_y_txt': None,
                      'id2token': train_data.id2agentAct,
                      'prefix': act_prefix,
                      'task_name': act_task_name,
                      'model_folder': argparams['model_folder']}
        act_model = MultiLabelClassifier(**act_kwargs)
        act_model.load_model(act_model_fname)
        act_X_vecBin = np.hstack((userTag_vecBin_pred, userIntent_pred_indicator))
        act_pred_indicator = act_model.predict(act_X_vecBin, test_data.agentAct_vecBin, test_data.userUtter_txt, test_data.agentAct_txt)
