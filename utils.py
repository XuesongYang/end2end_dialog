''' Description: Frequently used functions

    Author      : Xuesong Yang
    Email       : xyang45@illinois.edu
    Created Date: Dec. 31, 2016
'''

import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams.update({'figure.autolayout': True,
                 'font.size': 10, 'legend.fontsize': 10})


def checkExistence(data_path):
    ''' verify the existence of the data file
    '''
    if not os.path.exists(data_path):
        raise Exception('File/Folder not found: {}'.format(data_path))


def eval_slotTagging(tag_probs, mask_array, tag_trueLabel, O_id):
    ''' Evaluation for slot tagging.
        
        Inputs:
            tag_probs: shape = (sample_nb, maxlen_userUtter, tag_vocab_size), predicted probs
            mask_array: shape = (sample_nb, maxlen_userUtter), mask array with 0s for padding.
            tag_trueLabel: shape is the same to tag_probs, indicator sparse matrix. If all zeros in one sample, the padding is assumed.
            id2tag: dict of id to tag string
            conll_fname: file name of .conll format that is suitable for conlleval.pl as input
        Outputs: 
            precision, recall, and f1_score at token level using conlleval.pl, FYI, 'O' is not counted as a token.
            accuracy at frame level.
    '''
    pred_tag_ids_masked = (np.argmax(tag_probs, axis=-1) + 1) * mask_array
    true_tag_ids_masked = (np.argmax(tag_trueLabel, axis=-1) + 1) * mask_array
    pred_tag_ids_noO = np.array(pred_tag_ids_masked)
    true_tag_ids_noO = np.array(true_tag_ids_masked)
    pred_tag_ids_noO[pred_tag_ids_masked == O_id] = 0  # exclude 'O' token
    true_tag_ids_noO[true_tag_ids_masked == O_id] = 0  # exclude 'O' token
    nb_classes = tag_probs.shape[-1]
    pred_tag_1hot_noO = to_categorical(pred_tag_ids_noO, nb_classes)
    true_tag_1hot_noO = to_categorical(true_tag_ids_noO, nb_classes)
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_tag_1hot_noO.ravel(), pred_tag_1hot_noO.ravel(), beta=1.0, pos_label=1, average='binary')
    # true_tag_ids = (np.argmax(tag_trueLabel, axis=-1) + 1) * mask_array
    # [Q] Does all 'O's count in the denominator? temporally it is counted.
    accuracy_frame = calculate_FrameAccuracy(
        pred_tag_ids_masked, true_tag_ids_masked)
    return (precision, recall, f1_score, accuracy_frame)


def eval_actPred(act_probs, act_trueLabel, threshold):
    ''' make decision of label prediction given specific threshold
    '''
    preds_indicator = np.zeros_like(act_probs)
    preds_indicator[act_probs >= threshold] = 1
    precision, recall, f1_score, _ = precision_recall_fscore_support(act_trueLabel.ravel(), preds_indicator.ravel(), beta=1.0, pos_label=1, average='binary')
    accuracy_frame = calculate_FrameAccuracy(preds_indicator, act_trueLabel)
    return (preds_indicator, precision, recall, f1_score, accuracy_frame)


def to_categorical(y_seq, nb_classes):
    ''' transform into a 1hot matrix
        Input:
            y_seq: shape = (sample_nb, maxlen_userUtter), elements are token ids.
            nb_classes: scalar, tag_vocab_size
        Output:
            Y: shape = (sample_nb, maxlen_userUtter, tag_vocab_size)
    '''
    Y = np.zeros(y_seq.shape + (nb_classes,))
    for sample_idx, sample in enumerate(y_seq):
        for tag_idx, tag in enumerate(sample):
            # 0 denotes zero pads, which is not considered as one of class
            # labels.
            if tag != 0:
                Y[sample_idx, tag_idx, int(tag) - 1] = 1
    return Y


def calculate_FrameAccuracy(pred, true):
    ''' calculate frame-level accuracy = hit / sample_nb
            inputs:
                pred: shape = (sample_nb, dim_size), predicted ids matrix
                true: shape is the same to pred, true ids matrix
            Outputs:
                accuracy_frame
    '''
    compare_array = np.all((pred - true) == 0, axis=-1)
    hit = np.sum(compare_array.astype(int))
    sample_nb = true.shape[0]
    accuracy_frame = hit * 1. / sample_nb
    return accuracy_frame


def eval_intentPredict(intent_probs, intent_trueLabel):
    ''' Inputs:
            intent_probs: shape = (sample_nb, intent_vocab_size), predicted probs for intent prediction
            intent_trueLabel: shape = (sample_nb, intent_vocab_size), target binary matrix
        Output:
            precision, recall, f1_score, and threshold (prob >= threshold)
                        frame level accuracy
    '''
    # exclude the last element in precision and recall
    # which denotes 0 recall, and 1 precision
    precision, recall, thresholds = precision_recall_curve(
        intent_trueLabel.ravel(), intent_probs.ravel(), pos_label=1)
    f1_score = 2. * precision * recall / (precision + recall)
    f1_score[np.isnan(f1_score)] = 0.
    max_idx = np.argmax(f1_score[:-1])
    indicator = np.zeros_like(intent_probs)
    indicator[intent_probs >= thresholds[max_idx]] = 1
    accuracy_frame = calculate_FrameAccuracy(indicator, intent_trueLabel)
    return (precision[max_idx], recall[max_idx], f1_score[max_idx], accuracy_frame, thresholds[max_idx])


def plotLossConverge(loss_dct, png_fname):
    ''' Plot learning curves of losses, and save it as .png file.
    
        Input: 
            loss_dct: {'train': [], 'val': [], 'train_tagging':[], 'train_intent': [],
                       'val_tagging': [], 'val_intent':[]}
    '''
    plt.figure()
    plt.title('Learning Curves')
    # plt.grid()
    for name, loss in loss_dct.iteritems():
        plt.plot(loss, linewidth=2, label=name)
    epoch_nb = len(loss)
    plt.xlim(xmax=epoch_nb - 1)
    plt.xticks(range(epoch_nb))
    # plt.ylim(ymax=max(train_array + valid_array))
    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(png_fname)


def print_params(params_dct):
    for key in sorted(params_dct.keys()):
        print('\t{} = {}'.format(key, params_dct[key]))


def get_windowedVec(vecs, window_size):
    ''' padding (window_size -1) zero vectors in front of vecs, and 
        chunking padded vecs iteratively using a sliding window. The
        chunking rate is 1.
        Inputs:
            vecs: shape = (sample_nb,) + vec_dims, a list of n-darray vector
            window_size: scalar, the size of sliding windowb
        Outputs:
            vecs_windowed: shape = (sample_nb, window_size) + vec_dims
    '''
    sample_nb = vecs.shape[0]
    vec_dims = vecs.shape[1:]
    zeros_vec = np.zeros((window_size - 1,) + vec_dims)
    vecs_zeropad = np.vstack((zeros_vec, vecs))
    vecs_windowed = np.zeros((sample_nb, window_size) + vec_dims)
    for sample_idx in xrange(sample_nb):
        start_idx = sample_idx
        end_idx = start_idx + window_size
        vecs_windowed[sample_idx] = vecs_zeropad[start_idx:end_idx]
    return vecs_windowed


def getNLUpred(tag_probs, tag_mask, id2tag, intent_probs, threshold, id2intent):
    tag_txt = list()
    pred_tag_ids = (np.argmax(tag_probs, axis=-1) + 1) * tag_mask
    for sample in pred_tag_ids:
        line_txt = [id2tag[tag_id].replace('tag-', '', 1)
                    for tag_id in sample if tag_id != 0]
        tag_txt.append(' '.join(line_txt))
    intent_txt = list()
    indicator = np.zeros_like(intent_probs)
    indicator[intent_probs >= threshold] = 1.
    for sample in indicator:
        if np.all(sample == 0):
            line_txt = ['null']
        else:
            line_txt = [id2intent[intent_id].replace(
                'intent-', '', 1) for intent_id in (np.flatnonzero(sample) + 1)]
        intent_txt.append(';'.join(line_txt))
    return np.asarray(tag_txt), np.asarray(intent_txt)


def getNLUframeAccuracy(tag_probs, tag_mask, tag_trueLabel, intent_probs, intent_trueLabel, threshold):
    pred_tag_ids = (np.argmax(tag_probs, axis=-1) + 1) * tag_mask
    indicator = np.zeros_like(intent_probs)
    indicator[intent_probs >= threshold] = 1
    pred = np.hstack((pred_tag_ids, indicator))
    true_tag_ids = (np.argmax(tag_trueLabel, axis=-1) + 1) * tag_mask
    true = np.hstack((true_tag_ids, intent_trueLabel))
    accuracy_frame = calculate_FrameAccuracy(pred, true)
    return accuracy_frame


def getTagPred(tag_probs, tag_mask, id2tag):
    tag_txt = list()
    pred_tag_ids = (np.argmax(tag_probs, axis=-1) + 1) * tag_mask
    for sample in pred_tag_ids:
        line_txt = [id2tag[tag_id].replace('tag-', '', 1) for tag_id in sample if tag_id != 0]
        tag_txt.append(' '.join(line_txt))
    return np.asarray(tag_txt)


def getActPred(act_probs, threshold, id2agentAct):
    ''' make agent action prediction according to the threshold
        Inputs:
            act_probs: shape = (sample_nb, act_vocab_size), predicted action probability
            threshold: scalar, well-tuned decision criterion
            id2agentAct: a dict of (id, act) pairs
        Outputs:
            act_txt: shape = (sample_nb,), textual prediction 
    '''
    act_txt = list()
    indicator = np.zeros_like(act_probs)
    indicator[act_probs >= threshold] = 1.
    for sample in indicator:
        if np.all(sample == 0):
            line_txt = ['null']
        else:
            line_txt = [id2agentAct[act_id].replace('act-', '', 1) for act_id in (np.flatnonzero(sample) + 1)]
        act_txt.append(';'.join(line_txt))
    return np.asarray(act_txt)


def writeTxt(data_lst, fname, prefix='', delimiter=None):
    with open(fname, 'wb') as f:
        for line in data_lst:
            line_lst = [token.replace(prefix, '') for token in line.strip().split(delimiter)]
            if delimiter is None:
                delimiter = ' '
            f.write('{}\n'.format(delimiter.join(line_lst)))
