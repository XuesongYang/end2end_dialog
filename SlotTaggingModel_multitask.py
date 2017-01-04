''' Natural language understanding model based on multi-task learning.
    This model is trained on two tasks: slot tagging and user intent prediction. 

    Inputs: user utterance, e.g. BOS w1 w2 ... EOS
    Outputs: slot tags and user intents, e.g. O O B-moviename ... O\tinform+moviename
    
    Author      : Xuesong Yang
    Email       : xyang45@illinois.edu
    Created Date: Dec. 31, 2016
'''
from DataSetCSVslotTagging import DataSetCSVslotTagging
from keras.layers import Input, LSTM, Dense, Dropout, merge, Embedding, TimeDistributed
from keras.models import Model
from utils import print_params, eval_slotTagging, eval_intentPredict, writeTxt, getNLUpred, getActPred, getTagPred, checkExistence, getNLUframeAccuracy, eval_actPred 
import os
import numpy as np
np.random.seed(1983)


def writeUtterTagIntentTxt(utter_txt, tag_txt, intent_txt, target_fname):
    with open(target_fname, 'wb') as f:
        for (utter, tag, intent) in zip(utter_txt, tag_txt, intent_txt):
            tag_new = [token.replace('tag-', '', 1) for token in tag.split()]
            intent_new = [token.replace('intent-', '', 1)
                          for token in intent.split(';')]
            new_line = '{}\t{}\t{}'.format(
                utter, ' '.join(tag_new), ';'.join(intent_new))
            f.write('{}\n'.format(new_line))


class SlotTaggingModel(object):

    def __init__(self, **argparams):
        self.train_data = argparams['train_data']
        if self.train_data is not None:
            assert isinstance(self.train_data, DataSetCSVslotTagging)
        self.test_data = argparams['test_data']
        if self.test_data is not None:
            assert isinstance(self.test_data, DataSetCSVslotTagging)
        self.dev_data = argparams['dev_data']
        if self.dev_data is not None:
            assert isinstance(self.dev_data, DataSetCSVslotTagging)
        self.model_folder = argparams['model_folder']
        if self.model_folder is None:
            pid = argparams['pid']
            self.model_folder = './model/slot_{}'.format(pid)
            os.makedirs('{}/weights'.format(self.model_folder))
            os.makedirs('{}/dev_results'.format(self.model_folder))
        self.epoch_nb = argparams['epoch_nb']
        self.batch_size = argparams['batch_size']
        self.embedding_size = argparams['embedding_size']
        self.hidden_size = argparams['hidden_size']
        self.dropout = argparams['dropout_ratio']
        self.optimizer = argparams['optimizer']
        self.patience = argparams['patience']
        self.loss = argparams['loss']
        self.test_tag_flag = argparams['test_tag_only']
        self.test_intent_flag = argparams['test_intent_only']
        self.threshold = argparams['threshold']
        self.weights_fname = argparams['weights_fname']
        self.params = argparams

    def _build(self):
        print('Building Graph ...')
        words_input = Input(shape=(self.maxlen_userUtter,),
                            dtype='int32', name='words_input')
        # reserve 0 for masking, therefore vocab_size + 1
        embeddings = Embedding(input_dim=self.word_vocab_size + 1,
                               output_dim=self.embedding_size,
                               input_length=self.maxlen_userUtter,
                               mask_zero=True)(words_input)
        embeddings = Dropout(self.dropout)(embeddings)
        lstm_forward = LSTM(output_dim=self.hidden_size,
                            return_sequences=True,
                            name='LSTM_forward')(embeddings)
        lstm_forward = Dropout(self.dropout)(lstm_forward)
        lstm_backward = LSTM(output_dim=self.hidden_size,
                             return_sequences=True,
                             go_backwards=True,
                             name='LSTM_backward')(embeddings)
        lstm_backward = Dropout(self.dropout)(lstm_backward)
        lstm_concat = merge([lstm_forward, lstm_backward],
                            mode='concat',
                            concat_axis=-1,
                            name='merge_bidirections')
        slot_softmax_seq = TimeDistributed(Dense(
            output_dim=self.userTag_vocab_size,
            activation='softmax'), name='slot_output')(lstm_concat)
        intent_summary = LSTM(output_dim=self.hidden_size,
                              return_sequences=False,
                              name='summarize_to_dense')(lstm_concat)
        intent_summary = Dropout(self.dropout)(intent_summary)
        # intent_softmax = Dense(output_dim=self.userIntent_vocab_size,
        # activation='softmax', name='intent_output')(intent_summary)
        intent_softmax = Dense(output_dim=self.userIntent_vocab_size,
                               activation='sigmoid', name='intent_output')(intent_summary)
        self.model = Model(input=words_input, output=[
                           slot_softmax_seq, intent_softmax])
        self.model.compile(optimizer=self.optimizer,
                           # metrics=['accuracy'],
                           sample_weight_mode={
                               'slot_output': 'temporal', 'intent_output': None},
                           loss={'slot_output': self.loss, 'intent_output': 'binary_crossentropy'})

    def train(self):
        print('Training model ...')
        # load params
        self.maxlen_userUtter = self.train_data.maxlen_userUtter
        self.word_vocab_size = self.train_data.word_vocab_size
        self.userIntent_vocab_size = self.train_data.userIntent_vocab_size
        self.userTag_vocab_size = self.train_data.userTag_vocab_size
        self.id2word = self.train_data.id2word
        self.id2userTag = self.train_data.id2userTag
        self.id2userIntent = self.train_data.id2userIntent
        self.userTag2id = self.train_data.userTag2id
        other_npz = '{}/other_vars.npz'.format(self.model_folder)
        train_vars = {'id2userTag': self.id2userTag,
                      'id2word': self.id2word,
                      'id2userIntent': self.id2userIntent,
                      'userTag2id': self.userTag2id,
                      'userTag_vocab_size': self.userTag_vocab_size,
                      'userIntent_vocab_size': self.userIntent_vocab_size,
                      'word_vocab_size': self.word_vocab_size,
                      'maxlen_userUtter': self.maxlen_userUtter}
        np.savez_compressed(other_npz, **train_vars)
        self.params['maxlen_userUtter'] = self.maxlen_userUtter
        self.params['word_vocab_size'] = self.word_vocab_size
        self.params['userTag_vocab_size'] = self.userTag_vocab_size
        self.params['userIntent_vocab_size'] = self.userIntent_vocab_size
        print_params(self.params)
        # build model graph, save graph and plot graph
        self._build()
        self._plot_graph()
        graph_yaml = '{}/graph-arch.yaml'.format(self.model_folder)
        with open(graph_yaml, 'w') as fyaml:
            fyaml.write(self.model.to_yaml())
        # load train data
        X_train = self.train_data.userUtter_encodePad
        tag_train = self.train_data.userTag_1hotPad
        intent_train = self.train_data.userIntent_vecBin
        train_utter_txt = self.train_data.userUtter_txt
        train_intent_txt = self.train_data.userIntent_txt
        train_tag_txt = self.train_data.userTag_txt
        train_target_fname = '{}/train.target'.format(self.model_folder)
        writeUtterTagIntentTxt(train_utter_txt, train_tag_txt, train_intent_txt, train_target_fname)
        # load dev data
        X_dev = self.dev_data.userUtter_encodePad
        tag_dev = self.dev_data.userTag_1hotPad
        intent_dev = self.dev_data.userIntent_vecBin
        dev_utter_txt = self.dev_data.userUtter_txt
        dev_intent_txt = self.dev_data.userIntent_txt
        dev_tag_txt = self.dev_data.userTag_txt
        dev_target_fname = '{}/dev.target'.format(self.model_folder)
        writeUtterTagIntentTxt(dev_utter_txt, dev_tag_txt, dev_intent_txt, dev_target_fname)
        # get mask matrix for train and dev set
        mask_array_train = np.zeros_like(X_train)
        mask_array_train[X_train != 0] = 1
        mask_array_dev = np.zeros_like(X_dev)
        mask_array_dev[X_dev != 0] = 1
        # jointly training
        for ep in xrange(self.epoch_nb):
            print('<Epoch {}>'.format(ep))
            self.model.fit(x=X_train,
                           y={'slot_output': tag_train,
                              'intent_output': intent_train},
                           sample_weight={'slot_output': mask_array_train,
                                          'intent_output': None},
                           batch_size=self.batch_size, nb_epoch=1, verbose=2)
            tag_probs, intent_probs = self.model.predict(X_dev)
            # calculate token-level scores
            precision_tag, recall_tag, fscore_tag, accuracy_frame_tag = eval_slotTagging(tag_probs, mask_array_dev,
                                                                                         tag_dev, self.userTag2id['tag-O'])
            print('SlotTagging: ep={}, precision={:.4f}, recall={:.4f}, fscore={:.4f}, accuracy_frame={:.4f}'.format(ep, precision_tag, recall_tag, fscore_tag, accuracy_frame_tag))
            precision_intent, recall_intent, fscore_intent, accuracy_frame_intent, threshold = eval_intentPredict(intent_probs,
                                                                                                                  intent_dev)
            print('Intent Prediction: ep={}, precision={:.4f}, recall={:.4f}, fscore={:.4f}, accuracy_frame={:.4f}, threshold={:.4f}'.format(ep, precision_intent, recall_intent, fscore_intent, accuracy_frame_intent, threshold))
            accuracy_frame_both = getNLUframeAccuracy(tag_probs, mask_array_dev, tag_dev, intent_probs, intent_dev, threshold)
            print('NLU Frame: ep={}, accuracy={:.4f}'.format(ep, accuracy_frame_both))
            dev_tag_pred_txt, dev_intent_pred_txt = getNLUpred(tag_probs, mask_array_dev, self.id2userTag, intent_probs, threshold, self.id2userIntent)
            dev_results_fname = '{}/dev_results/dev_ep={}.pred'.format(self.model_folder, ep)
            writeUtterTagIntentTxt(dev_utter_txt, dev_tag_pred_txt, dev_intent_pred_txt, dev_results_fname)
            print('Write dev results: {}'.format(dev_results_fname))
            weights_fname = '{}/weights/ep={}_tagF1={:.4f}frameAcc={:.4f}_intentF1={:.4f}frameAcc={:.4f}th={:.4f}.h5'.format(self.model_folder, ep, fscore_tag, accuracy_frame_tag, fscore_intent, accuracy_frame_intent, threshold)
            print('Saving Model: {}'.format(weights_fname))
            self.model.save_weights(weights_fname, overwrite=True)

    def _plot_graph(self):
        from keras.utils import visualize_util
        graph_png = '{}/graph-plot.png'.format(self.model_folder)
        visualize_util.plot(self.model,
                            to_file=graph_png,
                            show_shapes=True,
                            show_layer_names=True)

    def predict(self):
        print('Predicting ...')
        result_folder = '{}/test_result'.format(self.model_folder)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        # write user utters
        utter_fname = '{}/utter.txt'.format(result_folder)
        if not os.path.exists(utter_fname):
            utter_txt = self.test_data.userUtter_txt
            writeTxt(utter_txt, utter_fname, prefix='', delimiter=None)
        print('\ttest_utter={}'.format(utter_fname))
        # load test data and calculate posterior probs.
        X_test = self.test_data.userUtter_encodePad
        tag_probs, intent_probs = self.model.predict(X_test)  # a tuple, slot_tags and intents
        # make prediction
        if self.test_intent_flag:
            assert self.threshold is not None, 'Argument required: --threshold'
            intent_probs_fname = '{}/intentProb_{}.npz'.format(result_folder, os.path.basename(self.weights_fname).split('_')[0])
            np.savez_compressed(intent_probs_fname, probs=intent_probs)
            print('\tintent_probs={}'.format(intent_probs_fname))
            # write prediction test results
            pred_intent_fname = '{}/intent_{}.pred'.format(result_folder, os.path.basename(self.weights_fname).split('_')[0])
            pred_intent_txt = getActPred(intent_probs, self.threshold, self.id2userIntent)
            writeTxt(pred_intent_txt, pred_intent_fname, prefix='intent-', delimiter=';')
            print('\tintent_pred={}'.format(pred_intent_fname))
            # write target test
            target_intent_fname = '{}/intent_test.target'.format(result_folder)
            target_intent = self.test_data.userIntent_txt
            writeTxt(target_intent, target_intent_fname, prefix='intent-', delimiter=';')
            print('\tintent_target={}'.format(target_intent_fname))
            # calculate performance scores
            preds_indicator, precision, recall, fscore, accuracy_frame = eval_actPred(intent_probs,
                                                                                      self.test_data.userIntent_vecBin, 
                                                                                      self.threshold)
            print('IntentPred: precision={:.4f}, recall={:.4f}, fscore={:.4f}, accuracy_frame={:.4f}'.format(precision, recall, fscore, accuracy_frame))

        if self.test_tag_flag:
            tag_probs_fname = '{}/tagProb_{}.npz'.format(result_folder, os.path.basename(self.weights_fname).split('_')[0])
            np.savez_compressed(tag_probs_fname, probs=tag_probs)
            print('\ttag_probs={}'.format(tag_probs_fname))
            # write prediction results
            pred_tag_fname = '{}/tag_{}.pred'.format(result_folder, os.path.basename(self.weights_fname).split('_')[0])
            mask_test = np.zeros_like(X_test)
            mask_test[X_test != 0] = 1
            pred_tag_txt = getTagPred(tag_probs, mask_test, self.id2userTag)
            writeTxt(pred_tag_txt, pred_tag_fname, prefix='tag-', delimiter=None)
            print('\ttag_pred={}'.format(pred_tag_fname))
            # write target
            target_tag_fname = '{}/tag_test.target'.format(result_folder)
            target_tag = self.test_data.userTag_txt
            writeTxt(target_tag, target_tag_fname, prefix='tag-', delimiter=None)
            print('\ttag_target={}'.format(target_tag_fname))
            # calculate performance scores
            precision, recall, fscore, accuracy_frame = eval_slotTagging(tag_probs, mask_test,
                                                                         self.test_data.userTag_1hotPad, self.userTag2id['tag-O'])
            print('SlotTagging: precision={:.4f}, recall={:.4f}, fscore={:.4f}, accuracy_frame={:.4f}'.format(precision, recall, fscore, accuracy_frame))

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
        self.maxlen_userUtter = np.int32(npzfile['maxlen_userUtter'][()])
        self.word_vocab_size = np.int32(npzfile['word_vocab_size'][()])
        self.userTag_vocab_size = np.int32(npzfile['userTag_vocab_size'][()])
        self.userIntent_vocab_size = np.int32(
            npzfile['userIntent_vocab_size'][()])
        self.id2userTag = npzfile['id2userTag'][()]
        self.id2word = npzfile['id2word'][()]
        self.id2userIntent = npzfile['id2userIntent'][()]
        self.userTag2id = npzfile['userTag2id'][()]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-npz', dest='data_npz',
                        help='.npz file including instances of DataSetCSVslotTagging for train, dev and test')
    parser.add_argument('--loss', dest='loss',
                        default='categorical_crossentropy',
                        help='objective function')
    parser.add_argument('--optimizer', dest='optimizer',
                        default='adam', help='optimizer')
    parser.add_argument('--epoch-nb', dest='epoch_nb', type=int,
                        default=300, help='number of epoches')
    parser.add_argument('--embedding-size', dest='embedding_size', type=int,
                        default=512, help='the dimention of word embeddings.')
    parser.add_argument('--patience', dest='patience', type=int,
                        default=10, help='the patience for early stopping criteria')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        default=32, help='batch size')
    parser.add_argument('--hidden-size', dest='hidden_size', type=int,
                        default=128, help='the number of hidden units in recurrent layer')
    parser.add_argument('--dropout-ratio', dest='dropout_ratio',
                        type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--model-folder', dest='model_folder',
                        help='the folder contains graph.yaml, weights.h5, and other_vars.npz')
    parser.add_argument('--test-tag', dest='test_tag_only', action='store_true',
                        help='only perform user Tagging test if this option is activated.')
    parser.add_argument('--test-intent', dest='test_intent_only', action='store_true',
                        help='only perform user intent test if this option is activated.')
    parser.add_argument('--train', dest='train_only', action='store_true',
                        help='only perform training if this option is activated.')
    parser.add_argument('--weights-file', dest='weights_fname', help='.h5 weights file.')
    parser.add_argument('--threshold', dest='threshold', type=float, help='float number of threshold for multi-label prediction decision.')
    args = parser.parse_args()
    argparams = vars(args)
    # early stop criteria are different for two tasks, therefore one model is
    # chosen for each.
    test_tag_only = argparams['test_tag_only']
    test_intent_only = argparams['test_intent_only']
    train_only = argparams['train_only']
    assert train_only or test_tag_only or test_intent_only, 'Arguments required: either --train, --test-tag, or --test-intent'
    pid = os.getpid()
    argparams['pid'] = pid
    npz_fname = argparams['data_npz']
    checkExistence(npz_fname)
    data_npz = np.load(npz_fname)
    if train_only:  # train model
        argparams['train_data'] = data_npz['train_data'][()]
        argparams['dev_data'] = data_npz['dev_data'][()]
        argparams['test_data'] = None
        model = SlotTaggingModel(**argparams)
        model.train()
    else:
        # train_only is False, while test_only is True
        # need to load model
        argparams['train_data'] = None
        argparams['dev_data'] = None
        argparams['test_data'] = None
        if argparams['model_folder'] is None:
            raise Exception('Argument required: --model-folder')
        model = SlotTaggingModel(**argparams)
        model.load_model()
    # test
    if test_tag_only or test_intent_only:
        model.test_data = data_npz['test_data'][()]
        model.predict()
