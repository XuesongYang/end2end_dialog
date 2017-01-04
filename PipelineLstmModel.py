''' Pipelined bi-directional LSTM model.
    This model stacked biLSTM NLU and biLSTM SAP separate models together,
    and its weights are initilized by the ones of seprate models. Besides,
    for the SAP task, the decision threshold on the output layer is tuned 
    on dev data.
    
    Author      : Xuesong Yang
    Email       : xyang45@illinois.edu
    Created Date: Dec. 31, 2016
'''

import numpy as np
from utils import checkExistence, get_windowedVec, eval_intentPredict, getActPred
from AgentActClassifyingModel import writeUtterActTxt
from DataSetCSVagentActPred import DataSetCSVagentActPred
import os
import argparse


def load_model_NLU(model_weights, test_data):
    from SlotTaggingModel_multitask import SlotTaggingModel
    params = ['train_data', 'dev_data', 'epoch_nb', 'batch_size', 'embedding_size', 'hidden_size',
              'dropout_ratio', 'optimizer', 'patience', 'loss', 'test_tag_only', 'test_intent_only', 'threshold']
    argparams = {key: None for key in params}
    argparams['weights_fname'] = model_weights
    argparams['model_folder'] = os.path.dirname(model_weights).replace('/weights', '', 1)
    argparams['test_data'] = test_data
    model = SlotTaggingModel(**argparams)
    model.load_model()
    return model


#def load_model_Policy(model_weights, test_data, threshold):
def load_model_Policy(model_weights):
    from AgentActClassifyingModel import AgentActClassifying 
    params = ['train_data', 'dev_data', 'test_data', 'epoch_nb', 'batch_size', 'hidden_size',
              'dropout_ratio', 'optimizer', 'patience', 'loss', 'threshold']
    argparams = {key: None for key in params}
    argparams['weights_fname'] = model_weights
    argparams['model_folder'] = os.path.dirname(model_weights).replace('/weights', '', 1)
    argparams['threshold'] = 1.0 
#    argparams['test_data'] = test_data
    model = AgentActClassifying(**argparams)
    model.load_model()
    return model


def readTagPredTxt(tag_pred_txt, userTag2id, sample_nb, userTag_vocab_size):
    checkExistence(tag_pred_txt)
    indicator = np.zeros((sample_nb, userTag_vocab_size))
    with open(tag_pred_txt, 'rb') as f:
        for idx, line in enumerate(f):
            for tag in line.strip().split():
                tag = 'tag-{}'.format(tag)
                if tag in userTag2id:
                    pos = userTag2id[tag] - 1
                else:
                    pos = 0
                indicator[idx, pos] = 1.
    return indicator


def readIntentPredTxt(intent_pred_txt, userIntent2id, sample_nb, userIntent_vocab_size):
    checkExistence(intent_pred_txt)
    indicator = np.zeros((sample_nb, userIntent_vocab_size))
    with open(intent_pred_txt, 'rb') as f:
        for idx, line in enumerate(f):
            for intent in line.strip().split(';'):
                if intent == 'null':
                    continue
                intent = 'intent-{}'.format(intent)
                if intent in userIntent2id:
                    pos = userIntent2id[intent] - 1
                else:
                    pos = 0
                indicator[idx, pos] = 1.
    return indicator


def pipelinePrediction(test_data, tag_model_weights, intent_model_weights, act_model_weights, result_folder, tuneTh=True, threshold=None):
    # load slot tagging model, and make prediction
    tag_model = load_model_NLU(tag_model_weights, test_data)
    tag_model.test_tag_flag = True
    tag_model.model_folder = result_folder
    tag_model.predict() 
    tag_pred_txt = '{}/test_result/tag_{}.pred'.format(tag_model.model_folder, os.path.basename(tag_model_weights).split('_')[0])
    tag_pred_indicator = readTagPredTxt(tag_pred_txt, test_data.userTag2id,
                                        len(test_data.userTag_txt), test_data.userTag_vocab_size)

    # load user intent model and make prediction
    intent_model = load_model_NLU(intent_model_weights, test_data)
    intent_model.test_intent_flag = True
    intent_model.threshold = threshold_intent
    intent_model.model_folder = result_folder
    intent_model.predict() 
    intent_pred_txt = '{}/test_result/intent_{}.pred'.format(intent_model.model_folder, os.path.basename(intent_model_weights).split('_')[0])
    intent_pred_indicator = readIntentPredTxt(intent_pred_txt, test_data.userIntent2id,
                                              len(test_data.userIntent_txt), test_data.userIntent_vocab_size)

    # merge indicators of slot tagging and user intents, and generate windowed tagIntent matrix
    assert len(tag_pred_indicator) == len(intent_pred_indicator), 'sample_nb is not equal.'
    nlu_vecBin = np.hstack((tag_pred_indicator, intent_pred_indicator))
    
    # load agent act model and make prediction
    act_model = load_model_Policy(act_model_weights)
    act_model.model_folder = result_folder
    nlu_vecBin_windowed = get_windowedVec(nlu_vecBin, act_model.window_size)

    if tuneTh:
        # tune threshold
        print('Tuning threshold on Dev ...')
        act_probs = act_model.model.predict(nlu_vecBin_windowed)
        precision, recall, fscore, accuracy_frame, act_threshold = eval_intentPredict(act_probs, test_data.agentAct_vecBin)
        print('AgentActPred on Dev: precision={:.4f}, recall={:.4f}, fscore={:.4f}, accuracy_frame={:.4f}, threshold={:.4f}'.format(precision, recall, fscore, accuracy_frame, act_threshold))
        dev_pred_txt = getActPred(act_probs, act_threshold, test_data.id2agentAct)
        dev_results_fname = '{}/act_dev.pred'.format(act_model.model_folder)
        writeUtterActTxt(test_data.userUtter_txt, dev_pred_txt, dev_results_fname)
        print('Write dev results: {}'.format(dev_results_fname))
        return act_threshold
    else:
        # make prediction based on well-tuned threshold
        assert threshold is not None, 'Argument required: threshold for agent action prediction.'
        act_model.threshold = threshold
        act_model.test_data = test_data
        act_model.test_data.userTagIntent_vecBin = nlu_vecBin_windowed
        act_model.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-npz', dest='npz_file', help='.npz file that contains the instance of DataSetCSVagentAct class')
    parser.add_argument('--intent-weights', dest='intent_weights', help='.h5 weights for best user intent model')
    parser.add_argument('--tag-weights', dest='tag_weights', help='.h5 weights for best user slot tagging model')
    parser.add_argument('--act-weights', dest='act_weights', help='.h5 weights for oracle agent act model')
    parser.add_argument('--intent-threshold', dest='intent_threshold', type=float, help='decision threshold for intent model')
    parser.add_argument('--tune', dest='tune_threshold', action='store_true', help='tune decision threshold for act model if this option is activated.')
    parser.add_argument('--act-threshold', dest='act_threshold', type=float, help='decision threshold for agent act model')
    parser.add_argument('--model-folder', dest='model_folder', help='model folder')
    args = parser.parse_args()
    argparams = vars(args)
    pid = os.getpid()
    npz_file = argparams['npz_file']
    intent_model_weights = argparams['intent_weights']
    tag_model_weights = argparams['tag_weights']
    act_model_weights = argparams['act_weights']
    threshold_intent = argparams['intent_threshold']
    tune_threshold = argparams['tune_threshold']
    threshold_act = argparams['act_threshold']
     
    # validate params
    checkExistence(npz_file)
    checkExistence(intent_model_weights)
    checkExistence(tag_model_weights)
    checkExistence(act_model_weights)
    assert threshold_intent is not None, 'Argument required: --intent-threshold' 
    for key in sorted(argparams.keys()):
        print('\t{}={}'.format(key, argparams[key])) 
 
    # load test data
    data_npz = np.load(npz_file)
    
    if tune_threshold:
        dev_result_folder = './model/pipe_{}/dev'.format(pid) 
        if not os.path.exists(dev_result_folder):
            os.makedirs(dev_result_folder)
        print('\tdev_result_folder={}'.format(dev_result_folder))
        dev_data = data_npz['dev_data'][()]
        assert isinstance(dev_data, DataSetCSVagentActPred)
        act_threshold = pipelinePrediction(dev_data, tag_model_weights, intent_model_weights, act_model_weights, dev_result_folder, tuneTh=True)
    else:
        assert threshold_act is not None, 'Argument required: --act-threshold.' 
        assert argparams['model_folder'] is not None, 'Argument required: --model-folder'
        test_result_folder = '{}/test'.format(argparams['model_folder']) 
        if not os.path.exists(test_result_folder):
            os.makedirs(test_result_folder)
        print('\ttest_result_folder={}'.format(test_result_folder))
        test_data = data_npz['test_data'][()]
        assert isinstance(test_data, DataSetCSVagentActPred)
        pipelinePrediction(test_data, tag_model_weights, intent_model_weights, act_model_weights, test_result_folder, tuneTh=False, threshold=threshold_act)
