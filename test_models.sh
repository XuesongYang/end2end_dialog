#! /bin/bash

# Author      : Xuesong Yang
# Email       : xyang45@illinois.edu
# Created Date: Dec. 31, 2016

######################################################################
# test models: 
#   1. the value for each variable is selected based on 
#      your well trained models.
#   2. for the pipelined LSTM model (PipelineLstmModel.py), you need to
#      firstly fine tuning the decision threshold of system act prediction,
#      and make testing. 
#   3. the following values are assigned as a demo to reproduce 
#      the performance in our paper. 
######################################################################

# biLSTM NLU model: slot tagging
weights_tag='./model/slot_4768/weights/ep=14_tagF1=0.468frameAcc=0.757_intentF1=0.399frameAcc=0.329th=0.203.h5'
model_folder_nlu='./model/slot_4768'
nohup python SlotTaggingModel_multitask.py --data-npz ./data/npz/dstc4.all.w-intent.slotTagging.npz --test-tag --weights-file $weights_tag --model-folder $model_folder_nlu &>log/nohup_slotTagging_tag_test.log &

# biLSTM NLU model: user intent 
weights_intent='./model/slot_4768/weights/ep=196_tagF1=0.448frameAcc=0.759_intentF1=0.496frameAcc=0.419th=0.391.h5'
threshold_intent=0.391
nohup python SlotTaggingModel_multitask.py --data-npz ./data/npz/dstc4.all.w-intent.slotTagging.npz --test-intent --weights-file $weights_tag --model-folder $model_folder_nlu --threshold $threshold_intent &>log/nohup_slotTagging_intent_test.log &

# biLSTM Oracle policy model
weights_oracle_act='./model/agentAct_4769/weights/ep=139_f1=0.228_frameAcc=0.202_th=0.154.h5'
model_folder_oracle_act='./model/agentAct_4769'
threshold_oracle_act=0.154
nohup python AgentActClassifyingModel.py --data-npz ./data/npz/dstc4.all.w-intent.agentActPred.npz --test --model-folder $model_folder_oracle_act --threshold $threshold_oracle_act --weights-file $weights_oracle_act &>log/nohup_agentAct_oracle_test.log & 

# biLSTM JointModel: slot tagging
weights_joint_tag='./model/joint_4770/weights/ep=8_tagF1=0.438_intentF1=0.494th=0.221_NLUframeAcc=0.296_actF1=0.302frameAcc=0.047th=0.131.h5'
model_folder_joint='./model/joint_4770'
nohup python JointModel_multitask_jointraining.py --data-npz ./data/npz/dstc4.all.w-intent.jointModel.npz --test-tag --model-folder $model_folder_joint --weights-file $weights_joint_tag &>log/nohup_jointModel_tag_test.log &

# biLSTM JointModel: user intent
weights_joint_intent='./model/joint_4770/weights/ep=13_tagF1=0.425_intentF1=0.519th=0.342_NLUframeAcc=0.379_actF1=0.300frameAcc=0.035th=0.139.h5'
threshold_joint_intent=0.342
nohup python JointModel_multitask_jointraining.py --data-npz ./data/npz/dstc4.all.w-intent.jointModel.npz --test-intent --model-folder $model_folder_joint --weights-file $weights_joint_intent --threshold $threshold_joint_intent &>log/nohup_jointModel_intent_test.log &

# biLSTM JointModel: agent act 
weights_joint_act='./model/joint_4770/weights/ep=172_tagF1=0.418_intentF1=0.492th=0.387_NLUframeAcc=0.360_actF1=0.189frameAcc=0.212th=0.009.h5'
threshold_joint_act=0.009
nohup python JointModel_multitask_jointraining.py --data-npz ./data/npz/dstc4.all.w-intent.jointModel.npz --test-act --model-folder $model_folder_joint --weights-file $weights_joint_act --threshold $threshold_joint_act &>log/nohup_jointModel_act_test.log &

# BaselineModel: CRFtagger for tagging, SVMs for intent, SVMs for policy
model_folder_baseline='./model/baseline_4771'
nohup python BaselineModel.py --data-npz ./data/npz/dstc4.all.w-intent.agentActPred.npz --test --model-folder $model_folder_baseline &>log/nohup_baseline_test.log &

# don't execute the next command until subshells finish.
wait

# [Optional] pipelined biLSTM model: the following commented script is used to tune the best threshold for act prediction on dev. We need to get this threshold before making prediction on test data.
# nohup python PipelineLstmModel.py --data-npz ./data/npz/dstc4.all.w-intent.agentActPred.npz --intent-weights $weights_intent --intent-threshold $threshold_intent --tag-weights $weights_tag --act-weights $weights_oracle_act --tune &>log/nohup_pipelineBiLSTM_tune.log & 

# pipelined biLSTM model: feed in well-tuned act prediction threshold, and make prediction on test.
threshold_act_pipe=0.064
nohup python PipelineLstmModel.py --data-npz ./data/npz/dstc4.all.w-intent.agentActPred.npz --intent-weights $weights_intent --intent-threshold $threshold_intent --tag-weights $weights_tag --act-weights $weights_oracle_act --act-threshold $threshold_act_pipe --model-folder ./model/pipe_30857 &>log/nohup_pipelineBiLSTM_test.log& 

# don't execute the next command until subshells finish.
wait

# evaluation table scores
python analyze_log.py
