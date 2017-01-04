# -*- coding: UTF-8 -*-
''' convert original .iob file into .csv file
        a) multiple actions are merged into a single string with delimiter ';';
        b) multiple intents are merged into a single string with delimiter ';';
        c) intents is defined as the user action;
        d) act is defined as the agent action.
    Output Fieldnames:
        Episode_ID, User_Utter, User_SlotTags, User_Intents, 
        Agent_Utter, Agent_PrevAct, Agent_Act
'''
from utils import checkExistence
import os
import re
import json
import glob
import csv
import argparse
from collections import defaultdict
from nltk.tokenize import word_tokenize


def mergeGuide(dct_lst):
    output_lst = list()
    line_prev = dct_lst[0]
    for line_dct in dct_lst[1:]:
        episode_id = int(line_dct['Episode_ID'])
        turn_id = int(line_dct['Turn_ID'])
        speaker = line_dct['Speaker']
        if episode_id == int(line_prev['Episode_ID']):
            if speaker == line_prev['Speaker'] and speaker == 'Guide':
                turn_id = int(line_dct['Turn_ID'])
                utter = '{}\t{}'.format(
                    line_prev['Utter'], line_dct['Utter'].strip())
                intent = '{};{}'.format(
                    line_prev['Intent'], line_dct['Intent'].strip())
                slot_tags = '{}\t{}'.format(
                    line_prev['SlotTags'], line_dct['SlotTags'].strip())
                line_prev = {'Turn_ID': turn_id,
                             'Speaker': speaker,
                             'Episode_ID': episode_id,
                             'Utter': utter,
                             'Intent': intent,
                             'SlotTags': slot_tags}
            elif speaker == line_prev['Speaker'] and speaker == 'Tourist':
                output_lst.append(line_prev)
                line_prev = line_dct
            elif speaker != line_prev['Speaker'] and speaker == 'Guide':
                output_lst.append(line_prev)
                line_prev = line_dct
            elif speaker != line_prev['Speaker'] and speaker == 'Tourist':
                output_lst.append(line_prev)
                line_prev = line_dct
        else:
            output_lst.append(line_prev)
            line_prev = line_dct
    output_lst.append(line_prev)
    return output_lst


def mergeCSV(csv_dct_lst):
    dct_lst = mergeGuide(csv_dct_lst)
    output_lst = list()
    speaker_fisrtLine = dct_lst[0]['Speaker']
    speaker_secondLine = dct_lst[1]['Speaker']
    if speaker_fisrtLine == 'Tourist':
        act_prev = 'null'
        if speaker_secondLine == 'Guide':
            act = dct_lst[1]['Intent']
            agent_utter = dct_lst[1]['Utter']
        elif speaker_secondLine == 'Tourist':
            act = 'null'
            agent_utter = 'null'
        else:
            raise Exception('Unknown Speaker: {}'.format(speaker_secondLine))
        new_line = {'Episode_ID': int(dct_lst[0]['Episode_ID']),
                    'User_Utter': dct_lst[0]['Utter'],
                    'User_SlotTags': dct_lst[0]['SlotTags'],
                    'User_Intents': dct_lst[0]['Intent'],
                    'Agent_PrevAct': act_prev,
                    'Agent_Utter': agent_utter,
                    'Agent_Act': act}
        output_lst.append(new_line)

    for (line_prev, line_mid, line_post) in zip(dct_lst[:-2], dct_lst[1:-1], dct_lst[2:]):
        speaker_prev = line_prev['Speaker']
        speaker_mid = line_mid['Speaker']
        speaker_post = line_post['Speaker']
        if speaker_mid == 'Tourist':
            if speaker_prev == 'Guide':
                act_prev = line_prev['Intent']
            elif speaker_prev == 'Tourist':
                act_prev = 'null'
            else:
                raise Exception('Unknown Speaker: {}'.format(speaker_prev))
            if speaker_post == 'Guide':
                act = line_post['Intent']
                agent_utter = line_post['Utter']
            elif speaker_post == 'Tourist':
                act = 'null'
                agent_utter = 'null'
            else:
                raise Exception('Unknown Speaker: {}'.format(speaker_post))
            new_line = {'Episode_ID': int(line_mid['Episode_ID']),
                        'User_Utter': line_mid['Utter'],
                        'User_SlotTags': line_mid['SlotTags'],
                        'User_Intents': line_mid['Intent'],
                        'Agent_PrevAct': act_prev,
                        'Agent_Utter': agent_utter,
                        'Agent_Act': act}
            output_lst.append(new_line)
        elif speaker_mid == 'Guide':
            continue
        else:
            raise Exception('Unknown Speaker: {}'.format(speaker_mid))
    # processing the last line
    speaker_lastLine = dct_lst[-1]['Speaker']
    speaker_lastSecondLine = dct_lst[-2]['Speaker']
    if speaker_lastLine == 'Tourist':
        act = 'null'
        agent_utter = 'null'
        if speaker_lastSecondLine == 'Guide':
            act_prev = dct_lst[-2]['Intent']
        elif speaker_lastSecondLine == 'Tourist':
            act_prev = 'null'
        else:
            raise Exception('Unknown Speaker: {}'.format(
                speaker_lastSecondLine))
        new_line = {'Episode_ID': int(dct_lst[-1]['Episode_ID']),
                    'User_Utter': dct_lst[-1]['Utter'],
                    'User_SlotTags': dct_lst[-1]['SlotTags'],
                    'User_Intents': dct_lst[-1]['Intent'],
                    'Agent_PrevAct': act_prev,
                    'Agent_Utter': agent_utter,
                    'Agent_Act': act}
        output_lst.append(new_line)
    filednames = ['Episode_ID', 'User_Utter', 'User_SlotTags',
                  'User_Intents', 'Agent_PrevAct', 'Agent_Utter', 'Agent_Act']
    return output_lst, filednames


def writeCSV(dct_lst, fname, fieldnames=None):
    with open(fname, 'w') as fcsv:
        if fieldnames is None:
            fieldnames = dct_lst[0].keys()
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dct_lst)


def readIOB(iob_file, utter_search):
    with open(iob_file, 'rb') as fiob:
        dct_lst = list()
        conversation_id = 0
        message_id = 0
        speaker_prev = ''
        for line in fiob:
            speaker, utter_index, slotTags, intents, dialog_begin = line.strip().split('\t')

            if utter_index in utter_search:
                utter = utter_search[utter_index]
            else:
                raise Exception('utter_index not found: {}'.format(utter_index))

            if int(dialog_begin) == 1:
                conversation_id += 1

            if speaker != speaker_prev:
                message_id += 1
            speaker_prev = speaker
            new_line = {'Episode_ID': conversation_id,
                        'Turn_ID': message_id,
                        'Speaker': speaker.strip(),
                        'Utter': utter.strip(),
                        'Intent': ';'.join(intents.strip().split()),
                        'SlotTags': slotTags.strip()}
            dct_lst.append(new_line)
    return dct_lst


def transformLabelJson(root_subdialogs):
    ''' construct a search dictionary: {sudialogID_utterIndex: normalized_utterance}
    '''
    utter_index_dct = defaultdict()
    pattern = '<.+?>'
    flist = glob.glob(os.path.join(root_subdialogs, '*/label.json'))
    for fname in flist:
        subdialog_id = fname.rsplit('/', 2)[1]
        with open(fname, 'rb') as flabel:
            label_dct = json.load(flabel)
            for utter_dct in label_dct['utterances']:
                utter_index = utter_dct['utter_index']
                semantic_taged = ' '.join(utter_dct['semantic_tagged'])
                semantic_tagged_notag = re.sub(pattern, ' ', semantic_taged)
                transcript = ['BOS'] + word_tokenize(semantic_tagged_notag.lower()) + ['EOS']
                trans = ' '.join(transcript)
                new_id = '{}_{}'.format(subdialog_id, utter_index)
                utter_index_dct[new_id] = trans.encode('utf-8')
    return utter_index_dct


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--iob-file', dest='iob_file',
                        help='.iob file in DSTC4')
    parser.add_argument('--csv-file', dest='csv_file',
                        help='the path of converted .csv file in DSTC4')
    parser.add_argument('--root-subdialogs', dest='root_subdialogs',
                        help='the root directory of DSTC4 subdialogs.')
    args = parser.parse_args()
    iob_file = args.iob_file
    checkExistence(iob_file)
    csv_file = args.csv_file
    root_subdialogs = args.root_subdialogs
    checkExistence(root_subdialogs)
    
    utter_search = transformLabelJson(root_subdialogs)
    dct_lst = readIOB(iob_file, utter_search)
    output_lst, fieldnames = mergeCSV(dct_lst)
    writeCSV(output_lst, csv_file, fieldnames)
