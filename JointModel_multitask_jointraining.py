''' End-to-End Joint model of NLU and System Act Prediction. This joint model
    is trained in a way of multi-task learning with user intents prediction, 
    user slot tagging, and system act prediction.
    
    Author      : Xuesong Yang
    Email       : xyang45@illinois.edu
    Created Date: Dec. 31, 2016
'''
from DataSetCSVjoint import DataSetCSVjoint
import os
import numpy as np
from utils import print_params, writeTxt, eval_intentPredict, eval_slotTagging, getNLUframeAccuracy, getNLUpred, getActPred, getTagPred, checkExistence, eval_actPred 
from keras.models import Model
from keras.layers import Input, LSTM, merge, Dense, TimeDistributed, Dropout, Embedding


class JointModel(object):

    def __init__(self, **argparams):
        self.train_data = argparams['train_data']
        if self.train_data is not None:
            assert isinstance(self.train_data, DataSetCSVjoint)
        self.test_data = argparams['test_data']
        if self.test_data is not None:
            assert isinstance(self.test_data, DataSetCSVjoint)
        self.dev_data = argparams['dev_data']
        if self.dev_data is not None:
            assert isinstance(self.dev_data, DataSetCSVjoint)
        self.model_folder = argparams['model_folder']
        if self.model_folder is None:
            pid = argparams['pid']
            self.model_folder = './model/joint_{}'.format(pid)
            os.makedirs('{}/weights'.format(self.model_folder))
            os.makedirs('{}/dev_results'.format(self.model_folder))
        self.epoch_nb = argparams['epoch_nb']
        self.dropout = argparams['dropout_ratio']
        self.optimizer = argparams['optimizer']
        self.patience = argparams['patience']
        self.loss = argparams['loss']
        self.hidden_size = argparams['hidden_size']
        self.context_size = argparams['context_size']
        self.embedding_size = argparams['embedding_size']
        self.batch_size = argparams['batch_size']
        self.test_tag_flag = argparams['test_tag_only']
        self.test_act_flag = argparams['test_act_only']
        self.test_intent_flag = argparams['test_intent_only']
        self.threshold = argparams['threshold']
        self.weights_fname = argparams['weights_fname']
        self.params = argparams

    def _build(self):
        print('Building Graph ...')
        # NLU model
        words_input = Input(shape=(self.maxlen_userUtter,), dtype='int32', name='LU_input')
        # reserve 0 for masking, therefore vocab_size + 1
        embeddings = Embedding(input_dim=self.word_vocab_size + 1,
                               output_dim=self.embedding_size,
                               input_length=self.maxlen_userUtter,
                               mask_zero=True)(words_input)
        embeddings = Dropout(self.dropout)(embeddings)
        lstm_forward = LSTM(output_dim=self.hidden_size,
                            return_sequences=True,
                            name='LU_LSTM_forward')(embeddings)
        lstm_forward = Dropout(self.dropout)(lstm_forward)
        lstm_backward = LSTM(output_dim=self.hidden_size,
                             return_sequences=True,
                             go_backwards=True,
                             name='LU_LSTM_backward')(embeddings)
        lstm_backward = Dropout(self.dropout)(lstm_backward)
        lstm_concat = merge([lstm_forward, lstm_backward],
                            mode='concat',
                            concat_axis=-1,
                            name='LU_merge_bidirection')
        intent_recurrent = LSTM(output_dim=self.hidden_size, name='intent_LSTM')(lstm_concat)
        joint_recurrent = LSTM(output_dim=self.context_size, name='joint_LSTM')(lstm_concat)
        # slot tagging task
        slot_softmax = TimeDistributed(Dense(output_dim=self.userTag_vocab_size, activation='softmax'), name='slotTagging_task')(lstm_concat)
        slotTagging_model = Model(input=words_input, output=slot_softmax)
        # intent multi-label classification
        intent_softmax = Dense(output_dim=self.userIntent_vocab_size, activation='sigmoid',
                               name='intent_output')(intent_recurrent)
        intent_model = Model(input=words_input, output=intent_softmax)
        # LU model
        lu_model = Model(input=words_input, output=joint_recurrent, name='LU_Model')
        # joint model over time
        utters_input = Input(shape=(self.window_size, self.maxlen_userUtter), dtype='int32', name='SAP_input')
        # import ipdb; ipdb.set_trace()
        encoded_utter_sequence = TimeDistributed(lu_model)(utters_input)
        slot_softmax_window = TimeDistributed(slotTagging_model, name='slot_output')(utters_input)
        intent_softmax_window = TimeDistributed(intent_model, name='intent_output')(utters_input)
        encoded_lstm_forward = LSTM(output_dim=self.hidden_size,
                                    return_sequences=False,
                                    name='SAP_LSTM_forward')(encoded_utter_sequence)
        encoded_lstm_forward = Dropout(self.dropout)(encoded_lstm_forward)
        encoded_lstm_backward = LSTM(output_dim=self.hidden_size,
                                     return_sequences=False,
                                     go_backwards=True,
                                     name='SAP_LSTM_backward')(encoded_utter_sequence)
        encoded_lstm_backward = Dropout(self.dropout)(encoded_lstm_backward)
        encoded_lstm_merge = merge([encoded_lstm_forward, encoded_lstm_backward], mode='concat', concat_axis=-1, name='SAP_merge_bidirection')
        act_softmax = Dense(output_dim=self.agentAct_vocab_size,
                            activation='sigmoid', name='act_output')(encoded_lstm_merge)
        self.model = Model(input=utters_input, output=[slot_softmax_window, intent_softmax_window, act_softmax])
        self.model.compile(optimizer=self.optimizer,
                           loss={'slot_output': self.loss,
                                 'intent_output': 'binary_crossentropy',
                                 'act_output': 'binary_crossentropy'},
                           sample_weight_mode={'slot_output': 'temporal',
                                               'intent_output': 'temporal',
                                               'act_output': None})

    def train(self):
        print('Training model ...')
        self.maxlen_userUtter = self.train_data.maxlen_userUtter
        self.window_size = self.train_data.window_size
        self.word_vocab_size = self.train_data.word_vocab_size
        self.agentAct_vocab_size = self.train_data.agentAct_vocab_size
        self.userTag_vocab_size = self.train_data.userTag_vocab_size
        self.userIntent_vocab_size = self.train_data.userIntent_vocab_size
        self.id2agentAct = self.train_data.id2agentAct
        self.id2userIntent = self.train_data.id2userIntent
        self.id2userTag = self.train_data.id2userTag
        self.id2word = self.train_data.id2word
        self.userTag2id = self.train_data.userTag2id
        if self.context_size is None:
            self.context_size = self.train_data.userTagIntent_vocab_size
        other_npz = '{}/other_vars.npz'.format(self.model_folder)
        train_vars = {'id2agentAct': self.id2agentAct,
                      'id2userIntent': self.id2userIntent,
                      'id2word': self.id2word,
                      'id2userTag': self.id2userTag,
                      'agentAct_vocab_size': self.agentAct_vocab_size,
                      'userIntent_vocab_size': self.userIntent_vocab_size,
                      'userTag_vocab_size': self.userTag_vocab_size,
                      'word_vocab_size': self.word_vocab_size,
                      'maxlen_userUtter': self.maxlen_userUtter,
                      'window_size': self.window_size,
                      'userTag2id': self.userTag2id}
        np.savez_compressed(other_npz, **train_vars)
        self.params['maxlen_userUtter'] = self.maxlen_userUtter
        self.params['window_size'] = self.window_size
        self.params['word_vocab_size'] = self.word_vocab_size
        self.params['agentAct_vocab_size'] = self.agentAct_vocab_size
        self.params['userTag_vocab_size'] = self.userTag_vocab_size
        self.params['userIntent_vocab_size'] = self.userIntent_vocab_size
        print_params(self.params)
        # build model graph, save graph and plot graph
        self._build()
        self._plot_graph()
        graph_yaml = '{}/graph-arch.yaml'.format(self.model_folder)
        with open(graph_yaml, 'w') as fyaml:
            fyaml.write(self.model.to_yaml())
        # load training data
        X_train = self.train_data.userUtter_encodePad_window
        tag_train = self.train_data.userTag_1hotPad_window
        intent_train = self.train_data.userIntent_vecBin_window
        act_train = self.train_data.agentAct_vecBin
        train_utter_txt = self.train_data.userUtter_txt
        train_intent_txt = self.train_data.userIntent_txt
        train_tag_txt = self.train_data.userTag_txt
        train_act_txt = self.train_data.agentAct_txt
        train_utter_fname = '{}/utter_train.target'.format(self.model_folder)
        writeTxt(train_utter_txt, train_utter_fname, prefix='', delimiter=None)
        train_intent_fname = '{}/intent_train.target'.format(self.model_folder)
        writeTxt(train_intent_txt, train_intent_fname, prefix='intent-', delimiter=';')
        train_tag_fname = '{}/tag_train.target'.format(self.model_folder)
        writeTxt(train_tag_txt, train_tag_fname, prefix='tag-', delimiter=None)
        train_act_fname = '{}/act_train.target'.format(self.model_folder)
        writeTxt(train_act_txt, train_act_fname, prefix='act-', delimiter=';')
        # load dev data
        X_dev = self.dev_data.userUtter_encodePad_window
        tag_dev = self.dev_data.userTag_1hotPad_window
        intent_dev = self.dev_data.userIntent_vecBin_window
        act_dev = self.dev_data.agentAct_vecBin
        dev_utter_txt = self.dev_data.userUtter_txt
        dev_intent_txt = self.dev_data.userIntent_txt
        dev_tag_txt = self.dev_data.userTag_txt
        dev_act_txt = self.dev_data.agentAct_txt
        dev_utter_fname = '{}/utter_dev.target'.format(self.model_folder)
        writeTxt(dev_utter_txt, dev_utter_fname, prefix='', delimiter=None)
        dev_intent_fname = '{}/intent_dev.target'.format(self.model_folder)
        writeTxt(dev_intent_txt, dev_intent_fname, prefix='intent-', delimiter=';')
        dev_tag_fname = '{}/tag_dev.target'.format(self.model_folder)
        writeTxt(dev_tag_txt, dev_tag_fname, prefix='tag-', delimiter=None)
        dev_act_fname = '{}/act_dev.target'.format(self.model_folder)
        writeTxt(dev_act_txt, dev_act_fname, prefix='act-', delimiter=';')
        # get mask matrix for train and dev data
        mask_train = np.zeros((X_train.shape[0], X_train.shape[1]))
        mask_train[np.any(X_train != 0, axis=-1)] = 1
        mask_dev = np.zeros((X_dev.shape[0], X_dev.shape[1]))
        mask_dev[np.any(X_dev != 0, axis=-1)] = 1
        mask_dev_maxlen = np.zeros_like(X_dev[:, -1])
        mask_dev_maxlen[X_dev[:, -1] != 0] = 1
        # joint training
        for ep in xrange(self.epoch_nb):
            print('<Epoch {}>'.format(ep))
            self.model.fit(x=X_train,
                           y={'slot_output': tag_train,
                              'intent_output': intent_train,
                              'act_output': act_train},
                           sample_weight={'slot_output': mask_train,
                                          'intent_output': mask_train,
                                          'act_output': None},
                           batch_size=self.batch_size, nb_epoch=1, verbose=2)
            tag_probs, intent_probs, act_probs = self.model.predict(X_dev)
            # evaluation for agent act
            precision_act, recall_act, fscore_act, accuracy_frame_act, threshold_act = eval_intentPredict(act_probs, act_dev)
            print('Agent Act Prediction: ep={}, precision={:.4f}, recall={:.4f}, fscore={:.4f}, accuracy_frame={:.4f}, threshold={:.4f}'.format(ep, precision_act, recall_act, fscore_act, accuracy_frame_act, threshold_act))
            # evaluation for slot tags
            precision_tag, recall_tag, fscore_tag, accuracy_frame_tag = eval_slotTagging(
                tag_probs[:, -1], mask_dev_maxlen, tag_dev[:, -1], self.userTag2id['tag-O'])
            print('SlotTagging: ep={}, precision={:.4f}, recall={:.4f}, fscore={:.4f}, accuracy_frame={:.4f}'.format(ep, precision_tag, recall_tag, fscore_tag, accuracy_frame_tag))
            # evaluation for user intent
            precision_intent, recall_intent, fscore_intent, accuracy_frame_intent, threshold_intent = eval_intentPredict(intent_probs[:, -1], intent_dev[:, -1])
            print('Intent Prediction: ep={}, precision={:.4f}, recall={:.4f}, fscore={:.4f}, accuracy_frame={:.4f}, threshold={:.4f}'.format(ep, precision_intent, recall_intent, fscore_intent, accuracy_frame_intent, threshold_intent))
            # frame-level accuracy of NLU
            accuracy_frame_both = getNLUframeAccuracy(tag_probs[:, -1], mask_dev_maxlen, tag_dev[:, -1], intent_probs[:, -1], intent_dev[:, -1], threshold_intent)
            print('NLU Frame: ep={}, accuracy={:.4f}'.format(ep, accuracy_frame_both))
            # save predicted results
            dev_tag_pred_txt, dev_intent_pred_txt = getNLUpred(tag_probs[:, -1], mask_dev_maxlen, self.id2userTag, intent_probs[:, -1], threshold_intent, self.id2userIntent)
            dev_act_pred_txt = getActPred(act_probs, threshold_act, self.id2agentAct)
            dev_tag_pred_fname = '{}/dev_results/tag_ep={}.pred'.format(self.model_folder, ep)
            writeTxt(dev_tag_pred_txt, dev_tag_pred_fname, prefix='tag-', delimiter=None)
            dev_intent_pred_fname = '{}/dev_results/intent_ep={}.pred'.format(self.model_folder, ep)
            writeTxt(dev_intent_pred_txt, dev_intent_pred_fname, prefix='intent-', delimiter=';')
            dev_act_pred_fname = '{}/dev_results/act_ep={}.pred'.format(self.model_folder, ep)
            writeTxt(dev_act_pred_txt, dev_act_pred_fname, prefix='act-', delimiter=';')
            dev_utter_pred_fname = '{}/dev_results/utter.txt'.format(self.model_folder)
            writeTxt(dev_utter_txt, dev_utter_pred_fname, prefix='', delimiter=None)
            print('Write dev results: {}, {}, {}'.format(dev_utter_pred_fname, dev_act_pred_fname, dev_tag_pred_fname, dev_intent_pred_fname))
            weights_fname = '{}/weights/ep={}_tagF1={:.4f}_intentF1={:.4f}th={:.4f}_NLUframeAcc={:.4f}_actF1={:.4f}frameAcc={:.4f}th={:.4f}.h5'.format(
                self.model_folder, ep, fscore_tag, fscore_intent, threshold_intent, accuracy_frame_both, fscore_act, accuracy_frame_act, threshold_act)
            print('Saving Model: {}'.format(weights_fname))
            self.model.save_weights(weights_fname, overwrite=True)

    def predict(self):
        # only write the last userIntent and userTag for each windowed sample
        print('Predicting ...')
        result_folder = '{}/test_result'.format(self.model_folder)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        # write user utters
        utter_fname = '{}/utter.txt'.format(result_folder)
        if not os.path.exists(utter_fname):
            test_utter_txt = self.test_data.userUtter_txt
            writeTxt(test_utter_txt, utter_fname, prefix='', delimiter=None)
        print('\ttest_utter={}'.format(utter_fname))
        # load test data and calculate posterior probs.
        X_test = self.test_data.userUtter_encodePad_window
        tag_probs, intent_probs, act_probs = self.model.predict(X_test)
        # make prediction
        if self.test_act_flag:
            assert self.threshold is not None, 'Threshold for agentAct is required.'
            act_probs_fname = '{}/actProb_{}.npz'.format(result_folder, os.path.basename(self.weights_fname).split('_')[0])
            np.savez_compressed(act_probs_fname, probs=act_probs)
            print('\tact_probs={}'.format(act_probs_fname))
            pred_act_fname = '{}/act_{}.pred'.format(result_folder, os.path.basename(self.weights_fname).split('_')[0])
            pred_act_txt = getActPred(act_probs, self.threshold, self.id2agentAct)
            writeTxt(pred_act_txt, pred_act_fname, prefix='act-', delimiter=';')
            print('\tact_pred={}'.format(pred_act_fname))
            target_act_fname = '{}/act_test.target'.format(result_folder)
            target_act = self.test_data.agentAct_txt
            writeTxt(target_act, target_act_fname, prefix='act-', delimiter=';')
            print('\tact_target={}'.format(target_act_fname))
            # calculate performance scores
            _, precision, recall, fscore, accuracy_frame = eval_actPred(act_probs, self.test_data.agentAct_vecBin,
                                                                        self.threshold)
            print('AgentActPred: precision={:.4f}, recall={:.4f}, fscore={:.4f}, accuracy_frame={:.4f}'.format(precision, recall, fscore, accuracy_frame))
        if self.test_intent_flag:
            assert self.threshold is not None, 'Threshold for userIntent is required.'
            intent_probs_fname = '{}/intentProb_{}.npz'.format(result_folder, os.path.basename(self.weights_fname).split('_')[0])
            np.savez_compressed(intent_probs_fname, probs=intent_probs)
            print('\tintent_probs={}'.format(intent_probs_fname))
            pred_intent_fname = '{}/intent_{}.pred'.format(result_folder, os.path.basename(self.weights_fname).split('_')[0])
            pred_intent_txt = getActPred(intent_probs[:, -1], self.threshold, self.id2userIntent)
            writeTxt(pred_intent_txt, pred_intent_fname, prefix='intent-', delimiter=';')
            print('\tintent_pred={}'.format(pred_intent_fname))
            target_intent_fname = '{}/intent_test.target'.format(result_folder)
            target_intent = self.test_data.userIntent_txt
            writeTxt(target_intent, target_intent_fname, prefix='intent-', delimiter=';')
            print('\tintent_target={}'.format(target_intent_fname))
            # calculate performance scores
            _, precision, recall, fscore, accuracy_frame = eval_actPred(intent_probs[:, -1], self.test_data.userIntent_vecBin_window[:, -1], self.threshold)
            print('IntentPred: precision={:.4f}, recall={:.4f}, fscore={:.4f}, accuracy_frame={:.4f}'.format(precision, recall, fscore, accuracy_frame))
        if self.test_tag_flag:
            tag_probs_fname = '{}/tagProb_{}.npz'.format(result_folder, os.path.basename(self.weights_fname).split('_')[0])
            np.savez_compressed(tag_probs_fname, probs=tag_probs)
            print('\ttag_probs={}'.format(tag_probs_fname))
            pred_tag_fname = '{}/tag_{}.pred'.format(result_folder, os.path.basename(self.weights_fname).split('_')[0])
            mask_test = np.zeros_like(X_test[:, -1])
            mask_test[X_test[:, -1] != 0] = 1
            pred_tag_txt = getTagPred(tag_probs[:, -1], mask_test, self.id2userTag)
            writeTxt(pred_tag_txt, pred_tag_fname, prefix='tag-', delimiter=None)
            print('\ttag_pred={}'.format(pred_tag_fname))
            target_tag_fname = '{}/tag_test.target'.format(result_folder)
            target_tag = self.test_data.userTag_txt
            writeTxt(target_tag, target_tag_fname, prefix='tag-', delimiter=None)
            print('\ttag_target={}'.format(target_tag_fname))
            # calculate performance scores
            precision, recall, fscore, accuracy_frame = eval_slotTagging(tag_probs[:, -1], mask_test, self.test_data.userTag_1hotPad_window[:, -1], self.userTag2id['tag-O'])
            print('SlotTagging: precision={:.4f}, recall={:.4f}, fscore={:.4f}, accuracy_frame={:.4f}'.format(precision, recall, fscore, accuracy_frame))

    def _plot_graph(self):
        from keras.utils import visualize_util
        graph_png = '{}/graph-plot.png'.format(self.model_folder)
        visualize_util.plot(self.model,
                            to_file=graph_png,
                            show_shapes=True,
                            show_layer_names=True)

    def load_model(self):
        print('Loading model ...')
        # check existence of params
        assert os.path.exists(self.model_folder), 'model_fold is not found: {}'.format(self.model_folder)
        assert self.weights_fname is not None, 'Argument required: --weights-file'
        checkExistence(self.weights_fname)
        model_graph = '{}/graph-arch.yaml'.format(self.model_folder)
        model_train_vars = '{}/other_vars.npz'.format(self.model_folder)
        checkExistence(model_graph)
        checkExistence(model_train_vars)
        from keras.models import model_from_yaml
        with open(model_graph, 'r') as fgraph:
            self.model = model_from_yaml(fgraph.read())
            self.model.load_weights(self.weights_fname)
        npzfile = np.load(model_train_vars)
        self.id2agentAct = npzfile['id2agentAct'][()]
        self.id2word = npzfile['id2word'][()]
        self.id2userTag = npzfile['id2userTag'][()]
        self.userTag2id = npzfile['userTag2id'][()]
        self.id2userIntent = npzfile['id2userIntent'][()]
        self.agentAct_vocab_size = np.int32(npzfile['agentAct_vocab_size'][()])
        self.userIntent_vocab_size = np.int32(npzfile['userIntent_vocab_size'][()])
        self.userTag_vocab_size = np.int32(npzfile['userTag_vocab_size'][()])
        self.word_vocab_size = np.int32(npzfile['word_vocab_size'][()])
        self.maxlen_userUtter = npzfile['maxlen_userUtter'][()]
        self.window_size = np.int32(npzfile['window_size'][()])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-npz', dest='data_npz',
                        help='.npz data file including train, dev and test')
    parser.add_argument('--loss', dest='loss',
                        default='categorical_crossentropy',
                        help='loss function')
    parser.add_argument('--optimizer', dest='optimizer',
                        default='adam', help='optimizer')
    parser.add_argument('--epoch-nb', dest='epoch_nb', type=int,
                        default=300, help='number of epoches')
    parser.add_argument('--patience', dest='patience', type=int,
                        default=10, help='patience for early stopping')
    parser.add_argument('--hidden-size', dest='hidden_size', type=int,
                        default=256, help='the number of hidden units in recurrent layer')
    parser.add_argument('--context-size', dest='context_size',
                        type=int, help='number of neurons in connection layer')
    parser.add_argument('--dropout-ratio', dest='dropout_ratio',
                        type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--model-folder', dest='model_folder',
                        help='the folder contains graph.yaml, weights.h5, and other_vars.npz, and results')
    parser.add_argument('--embedding-size', dest='embedding_size',
                        type=int, default=512, help='embed size')
    parser.add_argument('--batch-size', dest='batch_size',
                        type=int, default=32, help='batch size')
    parser.add_argument('--test-tag', dest='test_tag_only', action='store_true',
                        help='only perform user Tagging test if this option is activated.')
    parser.add_argument('--test-act', dest='test_act_only', action='store_true',
                        help='only perform agent act test if this option is activated.')
    parser.add_argument('--test-intent', dest='test_intent_only', action='store_true',
                        help='only perform user intent test if this option is activated.')
    parser.add_argument('--train', dest='train_only', action='store_true',
                        help='only perform training if this option is activated.')
    parser.add_argument('--weights-file', dest='weights_fname', help='.h5 weights file.')
    parser.add_argument('--threshold', dest='threshold', type=float, help='float number of threshold for multi-label prediction decision.')
    args = parser.parse_args()
    argparams = vars(args)
    pid = os.getpid()
    argparams['pid'] = pid
    # early stop criteria are different for two tasks, therefore one model is chosen for each.
    test_tag_only = argparams['test_tag_only']
    test_intent_only = argparams['test_intent_only']
    test_act_only = argparams['test_act_only']
    train_only = argparams['train_only']
    assert train_only or test_tag_only or test_intent_only or test_act_only, 'Arguments required: either --train, --test-tag, --test-intent, or --test-act'
    npz_fname = argparams['data_npz']
    checkExistence(npz_fname)
    data_npz = np.load(npz_fname)
    if train_only:  # train model
        argparams['train_data'] = data_npz['train_data'][()]
        argparams['dev_data'] = data_npz['dev_data'][()]
        argparams['test_data'] = None
        model = JointModel(**argparams)
        model.train()
    else:
        # train_only is False, while test_only is True
        # need to load model
        argparams['train_data'] = None
        argparams['dev_data'] = None
        argparams['test_data'] = None
        if argparams['model_folder'] is None:
            raise Exception('Argument required: --model-folder')
        model = JointModel(**argparams)
        model.load_model()
    # test
    if test_tag_only or test_act_only or test_intent_only:
        model.test_data = data_npz['test_data'][()]
        model.predict()
