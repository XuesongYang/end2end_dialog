''' Description : Extact performance scores in log folder, and print them in tables

    Author      : Xuesong Yang
    Email       : xyang45@illinois.edu
    Created Date: Dec.31, 2016
'''

import re
from prettytable import PrettyTable


def getScore(fname, prefix=''):
    score_lst = list()
    with open(fname, 'rb') as f:
        for line in f:
            m = re.match(r'{}precision=(?P<precision>.*), recall=(?P<recall>.*), fscore=(?P<fscore>.*), accuracy_frame=(?P<accuracy_frame>.*)'.format(prefix), line)
            if m is not None:
                score_dct = {key: val for (key, val) in m.groupdict().iteritems()}
                score_lst.append(score_dct)
    return score_lst


def baselineScore(fname):
    names_lst = ['oracle_act', 'nlu_tag', 'nlu_intent', 'policy']
    score_lst = getScore(fname, prefix='\t') 
    scores = {key: val for (key, val) in zip(names_lst, score_lst)}
    return scores


def jointModelScore(fname_tag, fname_intent, fname_act):
    names_lst = ['nlu_tag', 'nlu_intent', 'policy']
    score_tag = getScore(fname_tag, prefix='SlotTagging: ')
    score_intent = getScore(fname_intent, prefix='IntentPred: ')
    score_act = getScore(fname_act, prefix='AgentActPred: ')
    score_all = score_tag + score_intent + score_act
    scores = {key: val for (key, val) in zip(names_lst, score_all)}
    return scores


def slotTaggingScore(fname_tag, fname_intent):
    names_lst = ['nlu_tag', 'nlu_intent']
    score_tag = getScore(fname_tag, prefix='SlotTagging: ')
    score_intent = getScore(fname_intent, prefix='IntentPred: ')
    score_all = score_tag + score_intent
    scores = {key: val for (key, val) in zip(names_lst, score_all)}
    return scores


def bilstmOracleScore(fname):
    score_lst = getScore(fname, prefix='AgentActPred: ') 
    return {'policy': score_lst[0]}
   

def pipelineBilstmScore(fname):
    names_lst = ['nlu_tag', 'nlu_intent', 'policy']
    score_tag = getScore(fname, prefix='SlotTagging: ')
    score_intent = getScore(fname, prefix='IntentPred: ')
    score_act = getScore(fname, prefix='AgentActPred: ')
    score_all = score_tag + score_intent + score_act
    scores = {key: val for (key, val) in zip(names_lst, score_all)}
    return scores


def getFrameScore(tag_pred_fname, tag_target_fname, intent_pred_fname, intent_target_fname):
    hit = 0.
    sample_nb = 0.
    with open(tag_pred_fname, 'rb') as tag_fpred, open(tag_target_fname, 'rb') as tag_ftarget,\
            open(intent_pred_fname, 'rb') as intent_fpred, open(intent_target_fname, 'rb') as intent_ftarget:
        for (tag_pred, tag_target, intent_pred, intent_target) in zip(tag_fpred, tag_ftarget, intent_fpred, intent_ftarget):
            sample_nb += 1.
            i_pred = sorted(set(intent_pred.split(';')))
            i_target = sorted(set(intent_target.split(';')))
            if (i_pred == i_target) and (tag_pred == tag_target):
                hit += 1.
    accuracy_frame = hit / sample_nb
    return accuracy_frame


def nluFrameScore(baseline_fname, pipeline_fname, joint_tag_fname, joint_intent_fname):
    acc_frame = dict()
    baseline_tag_pred_fname, baseline_tag_target_fname, baseline_intent_pred_fname, baseline_intent_target_fname = getBaselineFnames(baseline_fname)
    baseline_AccFr = getFrameScore(baseline_tag_pred_fname, baseline_tag_target_fname, baseline_intent_pred_fname, baseline_intent_target_fname)
    acc_frame['Baseline'] = '{:.4f}'.format(baseline_AccFr)
    pipe_tag_pred_fname, pipe_tag_target_fname, pipe_intent_pred_fname, pipe_intent_target_fname = getPipelineFnames(pipeline_fname)
    pipeline_AccFr = getFrameScore(pipe_tag_pred_fname, pipe_tag_target_fname, pipe_intent_pred_fname, pipe_intent_target_fname)
    acc_frame['Pipeline'] = '{:.4f}'.format(pipeline_AccFr)
    joint_tag_pred_fname, joint_tag_target_fname = getFname(joint_tag_fname, prefix='tag')
    joint_intent_pred_fname, joint_intent_target_fname = getFname(joint_intent_fname, prefix='intent')
    joint_AccFr = getFrameScore(joint_tag_pred_fname, joint_tag_target_fname, joint_intent_pred_fname, joint_intent_target_fname)
    acc_frame['JointModel'] = '{:.4f}'.format(joint_AccFr)
    return acc_frame 

 
def getFname(fname, prefix=''):
    pred_fname = ''
    target_fname = ''
    with open(fname, 'rb') as f:
        for line in f:
            m = re.match(r'\t{0}_target=(?P<{0}_target>.*)'.format(prefix), line)
            if m is not None:
                target_fname = m.group('{}_target'.format(prefix)) 
                continue
            else:
                m = re.match(r'\t{0}_pred=(?P<{0}_pred>.*)'.format(prefix), line)
                if m is not None:
                    pred_fname = m.group('{}_pred'.format(prefix)) 
    assert target_fname != '' and pred_fname != '', 'Can not find file name.'
    return (pred_fname, target_fname)


def getPipelineFnames(pipeline_fname):
    tag_target_fname = ''
    tag_pred_fname = ''
    intent_target_fname = ''
    intent_pred_fname = ''
    with open(pipeline_fname, 'rb') as f:
        for line in f:
            m = re.match(r'\ttag_target=(?P<tag_target>.*)', line)
            if m is not None:
                tag_target_fname = m.group('tag_target') 
                continue
            else:
                m = re.match(r'\ttag_pred=(?P<tag_pred>.*)', line)
                if m is not None:
                    tag_pred_fname = m.group('tag_pred') 
                    continue
                else:
                    m = re.match(r'\tintent_pred=(?P<intent_pred>.*)', line)
                    if m is not None:
                        intent_pred_fname = m.group('intent_pred')
                        continue
                    else:
                        m = re.match(r'\tintent_target=(?P<intent_target>.*)', line)
                        if m is not None:
                            intent_target_fname = m.group('intent_target')
    assert tag_target_fname != '' and tag_pred_fname != '' and intent_target_fname != '' and intent_pred_fname != '', 'Can not find file name.'
    return (tag_pred_fname, tag_target_fname, intent_pred_fname, intent_target_fname)


def getBaselineFnames(baseline_fname):
    tag_target_fname = ''
    tag_pred_fname = ''
    intent_target_fname = ''
    intent_pred_fname = ''
    with open(baseline_fname, 'rb') as f:
        for line in f:
            m = re.match(r'\ttag_target=(?P<tag_target>.*)', line)
            if m is not None:
                tag_target_fname = m.group('tag_target') 
                continue
            else:
                m = re.match(r'\ttag_pred=(?P<tag_pred>.*)', line)
                if m is not None:
                    tag_pred_fname = m.group('tag_pred') 
                    continue
                else:
                    m = re.match(r'\ttest_pred=(?P<intent_pred>.*pipeline_intent-test.pred)', line)
                    if m is not None:
                        intent_pred_fname = m.group('intent_pred')
                        continue
                    else:
                        m = re.match(r'\ttest_target=(?P<intent_target>.*pipeline_intent-test.target)', line)
                        if m is not None:
                            intent_target_fname = m.group('intent_target')
    assert tag_target_fname != '' and tag_pred_fname != '' and intent_target_fname != '' and intent_pred_fname != '', 'Can not find file name.'
    return (tag_pred_fname, tag_target_fname, intent_pred_fname, intent_target_fname)


def tableEnd2End(baseline, pipeline, jointModel, bilstmOracle):
    table = PrettyTable()
    table.field_names = ['Models', 'Fscore', 'Precision', 'Recall', 'Accuracy_Frame']
    table.align['Models'] = 'l'
    table.add_row(['Baseline(CRF+SVMs+SVMs)', baseline['policy']['fscore'], baseline['policy']['precision'], baseline['policy']['recall'], baseline['policy']['accuracy_frame']])
    table.add_row(['Pipeline(biLSTM+biLSTM+biLSTM)', pipeline['policy']['fscore'], pipeline['policy']['precision'], pipeline['policy']['recall'], pipeline['policy']['accuracy_frame']])
    table.add_row(['JointModel(biLSTM+biLSTM+biLSTM)', jointModel['policy']['fscore'], jointModel['policy']['precision'], jointModel['policy']['recall'], jointModel['policy']['accuracy_frame']])
    table.add_row(['Oracle(SVMs)', baseline['oracle_act']['fscore'], baseline['oracle_act']['precision'], baseline['oracle_act']['recall'], baseline['oracle_act']['accuracy_frame']])
    table.add_row(['Oracle(biLSTM)', bilstmOracle['policy']['fscore'], bilstmOracle['policy']['precision'], bilstmOracle['policy']['recall'], bilstmOracle['policy']['accuracy_frame']])
    return table


def tableNLU(baseline, pipeline, jointModel, frame):
    table = PrettyTable()
    table.field_names = ['Models', 'tagF', 'tagP', 'tagR', 'tagAccFr', 'intF', 'intP', 'intR', 'intAccFr', 'nluAccFr']
    table.align['Models'] = 'l'
    table.add_row(['Baseline', baseline['nlu_tag']['fscore'], baseline['nlu_tag']['precision'], baseline['nlu_tag']['recall'], baseline['nlu_tag']['accuracy_frame'], baseline['nlu_intent']['fscore'], baseline['nlu_intent']['precision'], baseline['nlu_intent']['recall'], baseline['nlu_intent']['accuracy_frame'], frame['Baseline']])
    table.add_row(['Pipeline', pipeline['nlu_tag']['fscore'], pipeline['nlu_tag']['precision'], pipeline['nlu_tag']['recall'], pipeline['nlu_tag']['accuracy_frame'], pipeline['nlu_intent']['fscore'], pipeline['nlu_intent']['precision'], pipeline['nlu_intent']['recall'], pipeline['nlu_intent']['accuracy_frame'], frame['Pipeline']])
    table.add_row(['JointModel', jointModel['nlu_tag']['fscore'], jointModel['nlu_tag']['precision'], jointModel['nlu_tag']['recall'], jointModel['nlu_tag']['accuracy_frame'], jointModel['nlu_intent']['fscore'], jointModel['nlu_intent']['precision'], jointModel['nlu_intent']['recall'], jointModel['nlu_intent']['accuracy_frame'], frame['JointModel']])
    return table


if __name__ == '__main__':
    baseline = './log/nohup_baseline_test.log'
    pipelineBilstm = './log/nohup_pipelineBiLSTM_test.log'
    jointModel_tag = './log/nohup_jointModel_tag_test.log'
    jointModel_intent = './log/nohup_jointModel_intent_test.log'
    jointModel_act = './log/nohup_jointModel_act_test.log'
    slotTagging_intent = './log/nohup_slotTagging_intent_test.log'
    slotTagging_tag = './log/nohup_slotTagging_tag_test.log'
    agentAct_oracle = './log/nohup_agentAct_oracle_test.log'

    scores_baseline = baselineScore(baseline)
    # print(scores_baseline)

    scores_jointModel = jointModelScore(jointModel_tag, jointModel_intent, jointModel_act)
    # print(scores_jointModel)

    scores_slotTagging = slotTaggingScore(slotTagging_tag, slotTagging_intent)
    #print(scores_slotTagging)

    scores_bilstmOracle = bilstmOracleScore(agentAct_oracle)
    #print(scores_bilstmOracle)

    scores_pipelineBilstm = pipelineBilstmScore(pipelineBilstm)
    #print(scores_pipelineBilstm)
    
    scores_frame = nluFrameScore(baseline, pipelineBilstm, jointModel_tag, jointModel_intent)
    #print(scores_frame)

    end2end_table = tableEnd2End(scores_baseline, scores_pipelineBilstm, scores_jointModel, scores_bilstmOracle)
    print('Table 1 Perforamce of End2End Models')
    print(end2end_table)
    print('\n\n')

    nlu_table = tableNLU(scores_baseline, scores_pipelineBilstm, scores_jointModel, scores_frame)
    print('Table 2 Perforamce of NLU Models')
    print(nlu_table)
