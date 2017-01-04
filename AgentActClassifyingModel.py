''' Description: System action prediction oracle model.
        Inputs: binary vectors including slot_tags + user_intents
        Output: multi-label agent actions

    Author      : Xuesong Yang
    Email       : xyang45@illinois.edu
    Created Date: Dec. 31, 2016
'''

from DataSetCSVagentActPred import DataSetCSVagentActPred
from utils import print_params, checkExistence, getActPred, eval_intentPredict, eval_actPred 
from keras.layers import Input, LSTM, Dense, Dropout, merge
from keras.models import Model
import os
import numpy as np
np.random.seed(1983)


def writeUtterActTxt(utter_txt, act_txt, fname):
    with open(fname, 'wb') as f:
        for line_utter, line_act in zip(utter_txt, act_txt):
            line_lst = [token.replace('act-', '') for token in line_act.strip().split(';')]
            f.write('{}\t{}\n'.format(line_utter, ';'.join(line_lst)))


class AgentActClassifying(object):

    def __init__(self, **argparams):
        self.train_data = argparams['train_data']
        if self.train_data is not None:
            assert isinstance(self.train_data, DataSetCSVagentActPred)
        self.test_data = argparams['test_data']
        if self.test_data is not None:
            assert isinstance(self.test_data, DataSetCSVagentActPred)
        self.dev_data = argparams['dev_data']
        if self.dev_data is not None:
            assert isinstance(self.dev_data, DataSetCSVagentActPred)
        self.model_folder = argparams['model_folder']
        if self.model_folder is None:
            pid = argparams['pid']
            self.model_folder = './model/agentAct_{}'.format(pid)
            os.makedirs('{}/weights'.format(self.model_folder))
            os.makedirs('{}/dev_results'.format(self.model_folder))
        self.optimizer = argparams['optimizer']
        self.epoch_nb = argparams['epoch_nb']
        self.hidden_size = argparams['hidden_size']
        self.dropout = argparams['dropout_ratio']
        self.loss = argparams['loss']
        self.patience = argparams['patience']
        self.batch_size = argparams['batch_size']
        self.threshold = argparams['threshold']
        self.weights_fname = argparams['weights_fname']
        self.params = argparams

    def _build(self):
        print('Building Graph ...')
        inputs = Input(shape=(self.window_size, self.userTagIntent_vocab_size),
                       name='tagIntent_input')
        lstm_forward = LSTM(output_dim=self.hidden_size,
                            return_sequences=False,
                            name='LSTM_forward')(inputs)
        lstm_forward = Dropout(self.dropout)(lstm_forward)
        lstm_backward = LSTM(output_dim=self.hidden_size,
                             return_sequences=False,
                             go_backwards=True,
                             name='LSTM_backward')(inputs)
        lstm_backward = Dropout(self.dropout)(lstm_backward)
        lstm_concat = merge([lstm_forward, lstm_backward],
                            mode='concat', concat_axis=-1,
                            name='merge_bidirections')
        act_softmax = Dense(output_dim=self.agentAct_vocab_size,
                            activation='sigmoid')(lstm_concat)
        self.model = Model(input=inputs, output=act_softmax)
        self.model.compile(optimizer=self.optimizer,
                           loss='binary_crossentropy')

    def train(self):
        print('Training model ...')
        # load params
        self.window_size = self.train_data.window_size
        self.userTagIntent_vocab_size = self.train_data.userTagIntent_vocab_size
        self.agentAct_vocab_size = self.train_data.agentAct_vocab_size
        self.id2agentAct = self.train_data.id2agentAct
        other_npz = '{}/other_vars.npz'.format(self.model_folder)
        train_vars = {'window_size': self.window_size,
                      'userTagIntent_vocab_size': self.userTagIntent_vocab_size,
                      'agentAct_vocab_size': self.agentAct_vocab_size,
                      'id2agentAct': self.id2agentAct}
        np.savez_compressed(other_npz, **train_vars)
        self.params['window_size'] = self.window_size
        self.params['userTagIntent_vocab_size'] = self.userTagIntent_vocab_size
        self.params['agentAct_vocab_size'] = self.agentAct_vocab_size
        print_params(self.params)
        # build model graph, save graph and plot graph
        self._build()
        self._plot_graph()
        graph_yaml = '{}/graph-arch.yaml'.format(self.model_folder)
        with open(graph_yaml, 'w') as fyaml:
            fyaml.write(self.model.to_yaml())
        # load train data
        X_train = self.train_data.userTagIntent_vecBin
        y_train = self.train_data.agentAct_vecBin
        train_utter_txt = self.train_data.userUtter_txt
        train_act_txt = self.train_data.agentAct_txt
        train_fname = '{}/train.target'.format(self.model_folder)
        writeUtterActTxt(train_utter_txt, train_act_txt, train_fname)
        # load dev data
        X_dev = self.dev_data.userTagIntent_vecBin
        y_dev = self.dev_data.agentAct_vecBin
        dev_utter_txt = self.dev_data.userUtter_txt
        dev_act_txt = self.dev_data.agentAct_txt
        dev_fname = '{}/dev.target'.format(self.model_folder)
        writeUtterActTxt(dev_utter_txt, dev_act_txt, dev_fname)
        for ep in xrange(self.epoch_nb):
            print('<Epoch {}>'.format(ep))
            self.model.fit(x=X_train, y=y_train, batch_size=self.batch_size, nb_epoch=1, verbose=2)
            act_probs = self.model.predict(X_dev)
            precision, recall, fscore, accuracy_frame, threshold = eval_intentPredict(act_probs, y_dev)
            print('ep={}, precision={:.4f}, recall={:.4f}, fscore={:.4f}, accuracy_frame={:.4f}, threshold={:.4f}'.format(ep, precision, recall, fscore, accuracy_frame, threshold))
            dev_pred_txt = getActPred(act_probs, threshold, self.id2agentAct)
            dev_results_fname = '{}/dev_results/dev_ep={}.pred'.format(self.model_folder, ep)
            writeUtterActTxt(dev_utter_txt, dev_pred_txt, dev_results_fname)
            print('Write dev results: {}'.format(dev_results_fname))
            weights_fname = '{}/weights/ep={}_f1={:.4f}_frameAcc={:.4f}_th={:.4f}.h5'.format(self.model_folder, ep, fscore, accuracy_frame, threshold)
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
        result_folder = '{}/test_results'.format(self.model_folder)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        probs_fname = '{}/actProb_{}.npz'.format(result_folder, os.path.basename(self.weights_fname).split('_')[0])
        target_fname = '{}/act_test.target'.format(result_folder)
        pred_fname = '{}/act_{}.pred'.format(result_folder, os.path.basename(self.weights_fname).split('_')[0])
        print('\tact_probs={}'.format(probs_fname))
        print('\tact_target={}'.format(target_fname))
        print('\tact_pred={}'.format(pred_fname))
        utter_txt = self.test_data.userUtter_txt
        target_act = self.test_data.agentAct_txt
        writeUtterActTxt(utter_txt, target_act, target_fname)
        # prediction, save probs, and texts.
        X_test = self.test_data.userTagIntent_vecBin
        pred_probs = self.model.predict(X_test)
        np.savez_compressed(probs_fname, probs=pred_probs)
        pred_txt = getActPred(pred_probs, self.threshold, self.id2agentAct)
        writeUtterActTxt(utter_txt, pred_txt, pred_fname)
        # calculate performance scores
        _, precision, recall, fscore, accuracy_frame = eval_actPred(pred_probs, self.test_data.agentAct_vecBin,
                                                                    self.threshold)
        print('AgentActPred: precision={:.4f}, recall={:.4f}, fscore={:.4f}, accuracy_frame={:.4f}'.format(precision, recall, fscore, accuracy_frame))

    def load_model(self):
        print('Loading model ...')
        # check existence of params
        assert os.path.exists(self.model_folder), 'model_folder is not found: {}'.format(self.model_folder)
        assert self.threshold is not None, 'Argument required: --threshold'
        assert self.weights_fname is not None, 'Argument required: --weights-file'
        checkExistence(self.weights_fname)
        model_graph = '{}/graph-arch.yaml'.format(self.model_folder)
        model_train_vars = '{}/other_vars.npz'.format(self.model_folder)
        checkExistence(model_graph)
        checkExistence(model_train_vars)
        # load models
        from keras.models import model_from_yaml
        with open(model_graph, 'r') as fgraph:
            self.model = model_from_yaml(fgraph.read())
            self.model.load_weights(self.weights_fname)
        npzfile = np.load(model_train_vars)
        self.agentAct_vocab_size = np.int32(npzfile['agentAct_vocab_size'][()])
        self.userTagIntent_vocab_size = np.int32(npzfile['userTagIntent_vocab_size'][()])
        self.id2agentAct = npzfile['id2agentAct'][()]
        self.window_size = np.int32(npzfile['window_size'][()])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-npz', dest='data_npz',
                        help='.npz file including instances of DataSetCSVagentAct class for train, dev and test')
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
                        default=64, help='the number of hidden units in recurrent layer')
    parser.add_argument('--dropout-ratio', dest='dropout_ratio',
                        type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--model-folder', dest='model_folder',
                        help='the folder contains graph.yaml, weights.h5, and other_vars.npz, and results')
    parser.add_argument('--batch-size', dest='batch_size',
                        type=int, default=32, help='batch size')
    parser.add_argument('--test', dest='test_only', action='store_true',
                        help='only perform testing if this option is activated.')
    parser.add_argument('--train', dest='train_only', action='store_true',
                        help='only perform training if this option is activated.')
    parser.add_argument('--weights-file', dest='weights_fname', help='.h5 weights file.')
    parser.add_argument('--threshold', dest='threshold', type=float, help='float number of threshold for multi-label prediction decision.')
    args = parser.parse_args()
    argparams = vars(args)
    test_only = argparams['test_only']
    train_only = argparams['train_only']
    assert train_only or test_only, 'Arguments required: either --train or --test'
    pid = os.getpid()
    argparams['pid'] = pid
    npz_fname = argparams['data_npz']
    checkExistence(npz_fname)
    data_npz = np.load(npz_fname)
    if train_only:  # train model
        argparams['train_data'] = data_npz['train_data'][()]
        argparams['dev_data'] = data_npz['dev_data'][()]
        argparams['test_data'] = None
        model = AgentActClassifying(**argparams)
        model.train()
    else:
        # train_only is False, while test_only is True
        # need to load model
        argparams['train_data'] = None
        argparams['dev_data'] = None
        argparams['test_data'] = None
        if argparams['model_folder'] is None:
            raise Exception('Argument required: --model-folder')
        model = AgentActClassifying(**argparams)
        model.load_model()
    # test
    if test_only:
        test_data = data_npz['test_data'][()]
        model.test_data = test_data
        model.predict()
