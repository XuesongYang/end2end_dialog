#! /bin/bash
# Author      : Xuesong Yang
# Email       : xyang45@illinois.edu
# Created Date: Dec. 31, 2016


######################################################################
###################          prepare data     ########################
######################################################################

# convert .iob into .csv
root_subdialogs='/home/xyang45/work/end2end_dialog/data/DSTC5/data'
python iob2csv.py --iob-file ./data/iob/dstc4.all.w-intent.train.iob --csv-file ./data/csv/dstc4.all.w-intent.train.csv --root-subdialogs $root_subdialogs
python iob2csv.py --iob-file ./data/iob/dstc4.all.w-intent.dev.iob --csv-file ./data/csv/dstc4.all.w-intent.dev.csv --root-subdialogs $root_subdialogs
python iob2csv.py --iob-file ./data/iob/dstc4.all.w-intent.test.iob --csv-file ./data/csv/dstc4.all.w-intent.test.csv --root-subdialogs $root_subdialogs


# save instances of DataSetCSVslotTagging, DataSetCSVagentAct,
# DataSetCSVjoint, and DataSetCSVjoint_multitask
# into .npz files, or .pkl files if the file is too large.
python prepare_data.py


######################################################################
###################   train models            ########################
######################################################################

# hyperparams
epoch_nb=200

# NLU model: user utterance as input, user intent labels and slot tags as output
nohup python SlotTaggingModel_multitask.py --data-npz ./data/npz/dstc4.all.w-intent.slotTagging.npz --epoch-nb $epoch_nb --train &>log/nohup_slotTagging_train.log &

# Supervised policy model: human annotated slot tags and user intents as input, agent action labels as output
nohup python AgentActClassifyingModel.py --data-npz ./data/npz/dstc4.all.w-intent.agentActPred.npz --train --epoch-nb $epoch_nb &>log/nohup_agentAct_oracle_train.log &

# biLSTM based JointModel: windowed user utterance as input; user intent, slot tags and agent actions as output
nohup python JointModel_multitask_jointraining.py --data-npz ./data/npz/dstc4.all.w-intent.jointModel.npz --train --epoch-nb $epoch_nb &>log/nohup_jointModel_train.log & 

# BaselineModel: CRFtagger for slot tagging, OneVsRest Linear SVC for intent prediction, and OneVsRest Linear SVC for agent action
nohup python BaselineModel.py --data-npz ./data/npz/dstc4.all.w-intent.agentActPred.npz --train &>log/nohup_baseline_train.log &
