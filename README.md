# END-TO-END JOINT LEARNING OF NATURAL LANGUAGE UNDERSTANDING AND DIALOGUE MANAGER

This repository releases the source code for our paper [END-TO-END JOINT LEARNING OF NATURAL LANGUAGE UNDERSTANDING AND DIALOGUE MANAGER](https://arxiv.org/abs/1612.00913). Please cite the following paper if you use this code as part of any published research. 

    [1] X. Yang, Y. Chen, D. Hakkani-Tur, P. Crook, X. Li, J. Gao and L. Deng, "END-TO-END JOINT LEARNING OF NATURAL LANGUAGE UNDERSTANDING AND DIALOGUE MANAGER", in Proceedings of The 42nd IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017.

    @inproceedings{yang2016endtoend,
        title={END-TO-END JOINT LEARNING OF NATURAL LANGUAGE UNDERSTANDING AND DIALOGUE MANAGER},
        author={Yang, Xuesong and Chen, Yun-Nung and Hakkani-T\"ur, Dilek and Crook, Paul and Li, Xiujun and Gao, Jianfeng and Deng, Li},
        booktitle={Proceedings of The 42nd IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).},
        year={2017},
    }

## License

The code is released under [MIT License](./LICENSE).


## Data

We used DSTC4 data and split data into train/dev/test in the following:

| Split   | Total | Sub-Dialog IDs |
|:-------:|------:|:---------------|
| train   | 14    |001, 002, 003, 004, 006, 007, 008, 009, 010, 012, 013, 017, 019, 022 |
| dev     | 6     |011, 016, 020, 025, 026, 028 |
| test    | 9     |021, 023, 024, 030, 033, 035, 041, 047, 048 |

[Note]:

1. We are only allowed to provide user utterance index, IOB data, and their corresponding speech act and attributes. User raw utterances are not allowed to release. Please contact the committee of DSTC4 or DSTC5 for the whole data.

Utterance index consists of two components: sub-dialog folder and utter_id.

e.g.  `011_129` represents that the sub-dialog folder is 011, and utter_id is 129 in the corresponding `your_DSTC_directory/011/label.json` file.


2. Although we only mentioned DSTC4 in our paper, users can also locate these sub-dialogs from [DSTC5](http://workshop.colips.org/dstc5/data.html), since it provides all the same data in DSTC4, and plus two extra Chinese dialogs (055, 056).

## Prerequisites

1. `pip install nltk`
2. `pip install python-crfsuite`
3. `pip install prettytable`
4. `keras 1.2.0` 
5. `theano 0.9.0dev4`


## Executable scripts

1. training models: `$ bash train_models.sh`
2. testing models using selected parameters: `$ bash test_models.sh`


## Auxiliary Label 

1. "null" label for system actions.

    From human annotations, "null" label is used to identify that there is not any system action that makes response to current user utterance. In other word, "null" is not supposed to be one of the system actions. During the testing process, if the posterior prob for each oneVSall binary classifier is less than its decision threshold, "null" is considered as the predicted label.

2. "null" label for user intent.

    Similar explanation to the one for system actions.



## Model Selection

1. JointModel: 
    * slotTagging
        * weights=`./model/joint_4770/weights/ep=8_tagF1=0.438_intentF1=0.494th=0.221_NLUframeAcc=0.296_actF1=0.302frameAcc=0.047th=0.131.h5`
    * userIntent
        * weights=`./model/joint_4770/weights/ep=13_tagF1=0.425_intentF1=0.519th=0.342_NLUframeAcc=0.379_actF1=0.300frameAcc=0.035th=0.139.h5`
        * threshold=`0.342`
    * agentAct
        * weights=`./model/joint_4770/weights/ep=172_tagF1=0.418_intentF1=0.492th=0.387_NLUframeAcc=0.360_actF1=0.189frameAcc=0.212th=0.009.h5`
        * threshold=`0.009`

2. SlotTaggingModel:
    * slotTagging
        * weights=`./model/slot_4768/weights/ep=14_tagF1=0.468frameAcc=0.757_intentF1=0.399frameAcc=0.329th=0.203.h5`
    * userIntent
        * weights=`./model/slot_4768/weights/ep=196_tagF1=0.448frameAcc=0.759_intentF1=0.496frameAcc=0.419th=0.391.h5`
        * threshold=`0.391`

3. AgentActModel:
    * agentAct
        * weights=`./model/agentAct_4769/weights/ep=139_f1=0.228_frameAcc=0.202_th=0.154.h5`
        * threshold=`0.154`

4. BaselineModel: `model_folder=./model/baseline_4771`



## Performance

Table 1 Perforamce of End2End Models for System Act Prediction.

| Models                           | Fscore | Precision | Recall | Accuracy_Frame |
| :--------------------------------|--------|-----------|--------|----------------|
| Baseline(CRF+SVMs+SVMs)          | 0.3115 |   0.2992  | 0.3248 |     0.0771     |
| Pipeline(biLSTM+biLSTM+biLSTM)   | 0.1989 |   0.1487  | 0.3001 |     0.1196     |
| JointModel(biLSTM+biLSTM+biLSTM) | 0.1904 |   0.1853  | 0.1957 |     0.2284     |
| Oracle(SVMs)                     | 0.3061 |   0.3020  | 0.3104 |     0.0765     |
| Oracle(biLSTM)                   | 0.2309 |   0.2224  | 0.2401 |     0.1967     |



Table 2 Perforamce of NLU Models

| Models     |  tagF  |  tagP  |  tagR  | tagAccFr |  intF  |  intP  |  intR  | intAccFr | nluAccFr |
|:-----------|--------|--------|--------|----------|--------|--------|--------|----------|----------|
| Baseline   | 0.4050 | 0.6141 | 0.3021 |  0.7731  | 0.4975 | 0.5256 | 0.4724 |  0.3719  |  0.3313  |
| Pipeline   | 0.4615 | 0.5463 | 0.3996 |  0.7684  | 0.4748 | 0.5219 | 0.4355 |  0.3996  |  0.3638  |
| JointModel | 0.4504 | 0.5335 | 0.3897 |  0.7649  | 0.4967 | 0.5222 | 0.4735 |  0.4220  |  0.3738  |

