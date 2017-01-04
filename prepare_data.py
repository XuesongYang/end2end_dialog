''' Description: preparing data for experiments. 
                 train/dev/test are saved as .npz files, respectively.
    
    Author      : Xuesong Yang
    Email       : xyang45@illinois.edu
    Created Date: Dec. 31, 2016
'''
import numpy as np
import os


def prepare_slotTagging(csv_train, csv_test, csv_dev, npz_fname):
    from DataSetCSVslotTagging import DataSetCSVslotTagging
    train_data = DataSetCSVslotTagging(csv_train, flag='train')
    dev_data = DataSetCSVslotTagging(
        csv_dev, train_data=train_data, flag='test')
    test_data = DataSetCSVslotTagging(
        csv_test, train_data=train_data, flag='test')
    # get maxlen over all data: train, dev and test
    maxlen_userUtter_train = train_data.getUserUtterMaxlen()
    maxlen_userUtter_dev = dev_data.getUserUtterMaxlen()
    maxlen_userUtter_test = test_data.getUserUtterMaxlen()
    maxlen_userUtter = max(maxlen_userUtter_train, maxlen_userUtter_dev, maxlen_userUtter_test)
    train_data.transform_data(maxlen_userUtter)
    dev_data.transform_data(maxlen_userUtter)
    test_data.transform_data(maxlen_userUtter)
    # save .npz
    if os.path.exists(npz_fname):
        os.remove(npz_fname)
    np.savez_compressed(npz_fname, train_data=train_data, dev_data=dev_data, test_data=test_data)
 

def prepare_agentActPredict(csv_train, csv_test, csv_dev, npz_fname):
    from DataSetCSVagentActPred import DataSetCSVagentActPred
    train_data = DataSetCSVagentActPred(csv_train, window_size=5, flag='train') 
    dev_data = DataSetCSVagentActPred(csv_dev, train_data=train_data, flag='test')
    test_data = DataSetCSVagentActPred(csv_test, train_data=train_data, flag='test') 
    # get maxlen over all data: train, dev and test
    maxlen_userUtter_train = train_data.getUserUtterMaxlen()
    maxlen_userUtter_dev = dev_data.getUserUtterMaxlen()
    maxlen_userUtter_test = test_data.getUserUtterMaxlen()
    maxlen_userUtter = max(maxlen_userUtter_train, maxlen_userUtter_dev, maxlen_userUtter_test)
    train_data.transform_data(maxlen_userUtter)
    dev_data.transform_data(maxlen_userUtter)
    test_data.transform_data(maxlen_userUtter)
    # save .npz
    if os.path.exists(npz_fname):
        os.remove(npz_fname)
    np.savez_compressed(npz_fname, train_data=train_data, dev_data=dev_data, test_data=test_data)


def prepare_joint(csv_train, csv_test, csv_dev, npz_fname):
    from DataSetCSVjoint import DataSetCSVjoint
    train_data = DataSetCSVjoint(csv_file=csv_train, window_size=5, flag='train')
    dev_data = DataSetCSVjoint(csv_file=csv_dev, train_data=train_data, flag='test')
    test_data = DataSetCSVjoint(csv_file=csv_test, train_data=train_data, flag='test') 
    # get maxlen over all data: train, dev and test
    maxlen_userUtter_train = train_data.getUserUtterMaxlen()
    maxlen_userUtter_dev = dev_data.getUserUtterMaxlen()
    maxlen_userUtter_test = test_data.getUserUtterMaxlen()
    maxlen_userUtter = max(maxlen_userUtter_train, maxlen_userUtter_dev, maxlen_userUtter_test)
    train_data.transform_data(maxlen_userUtter)
    dev_data.transform_data(maxlen_userUtter)
    test_data.transform_data(maxlen_userUtter)
    # save .npz
    if os.path.exists(npz_fname):
        os.remove(npz_fname)
    np.savez_compressed(npz_fname, train_data=train_data, dev_data=dev_data, test_data=test_data)


if __name__ == '__main__':
    csv_train = './data/csv/dstc4.all.w-intent.train.csv'
    csv_test = './data/csv/dstc4.all.w-intent.test.csv'
    csv_dev = './data/csv/dstc4.all.w-intent.dev.csv'

    # process slot tagging data
    slot_npz = './data/npz/dstc4.all.w-intent.slotTagging.npz'
    prepare_slotTagging(csv_train, csv_test, csv_dev, slot_npz)
    # process agent act prediction data
    act_npz = './data/npz/dstc4.all.w-intent.agentActPred.npz'
    prepare_agentActPredict(csv_train, csv_test, csv_dev, act_npz)
    # process joint model data
    joint_npz = './data/npz/dstc4.all.w-intent.jointModel.npz'
    prepare_joint(csv_train, csv_test, csv_dev, joint_npz)
