# This code is modified by ceskalka for exploring a simple membership inference attack.
# See comments in main for relevant discussion.
import os
import random
import pickle
import numpy as np
import yaml

config_file = './env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def main():
    path_dir = os.path.join(root_dir, 'purchase')
    DATASET_PATH = os.path.join(path_dir, 'data')

    DATASET_FEATURES = os.path.join(DATASET_PATH,'X.npy')
    DATASET_LABELS = os.path.join(DATASET_PATH,'Y.npy')

    X = np.load(DATASET_FEATURES)
    Y = np.load(DATASET_LABELS)

    len_train =len(Y)
    if os.path.exists(os.path.join(DATASET_PATH,'random_r_purchase100')):
        r=pickle.load(open(os.path.join(DATASET_PATH,'random_r_purchase100'),'rb'))
    else:
        r = np.arange(len_train)
        np.random.shuffle(r)
        pickle.dump(r,open(os.path.join(DATASET_PATH,'random_r_purchase100'),'wb'))

    np.random.seed(0)
    X=X[r]
    Y=Y[r]
    r = np.arange(len_train)
    np.random.shuffle(r)
    X = X[r]
    Y = Y[r]

    train_ratio = .1
    test_ratio = .1

    # Victim training and test sets are both 10% of complete dataset and disjoint.
    train_v_start = 0
    train_v_end = int(train_ratio*len_train)
    test_v_start = train_v_end
    test_v_end = test_v_start + int(test_ratio*len_train)

    # Shadow training and test sets are both 10% of complete dataset and disjoint.
    # No overlap between any shadow and victim training or test data.
    train_s_start = test_v_end
    train_s_end = train_s_start + int(train_ratio*len_train)
    test_s_start = train_s_end
    test_s_end = test_s_start + int(test_ratio*len_train)
    
    train_data_v = X[train_v_start:train_v_end]
    train_label_v = Y[train_v_start:train_v_end]    
    test_data_v = X[test_v_start:test_v_end]
    test_label_v = Y[test_v_start:test_v_end]
    
    train_data_s = X[train_s_start:train_s_end]
    train_label_s = Y[train_s_start:train_s_end]    
    test_data_s = X[test_s_start:test_s_end]
    test_label_s = Y[test_s_start:test_s_end]

    path2 = os.path.join(DATASET_PATH, 'partition')
    if not os.path.isdir(path2):
        mkdir_p(path2)

    # Writing datasets to files. 
    np.save(os.path.join(DATASET_PATH, 'partition', 'train_data_v.npy'), train_data_v)
    np.save(os.path.join(DATASET_PATH, 'partition', 'train_label_v.npy'), train_label_v)
    np.save(os.path.join(DATASET_PATH, 'partition', 'test_data_v.npy'), test_data_v)
    np.save(os.path.join(DATASET_PATH, 'partition', 'test_label_v.npy'), test_label_v)
    np.save(os.path.join(DATASET_PATH, 'partition', 'train_data_s.npy'), train_data_s)
    np.save(os.path.join(DATASET_PATH, 'partition', 'train_label_s.npy'), train_label_s)
    np.save(os.path.join(DATASET_PATH, 'partition', 'test_data_s.npy'), test_data_s)
    np.save(os.path.join(DATASET_PATH, 'partition', 'test_label_s.npy'), test_label_s)

    print(DATASET_PATH)
    
if __name__ == '__main__':
    main()