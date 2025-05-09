# This code is modified by ceskalka for exploring a simple membership inference attack.
# See comments below including in attack_data and main functions for relevant discussion.
#
# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import random
import numpy as np
import sys
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

# Might need to cd into Undefend for this
config_file = './../../env.yml'

with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']

sys.path.append(src_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(src_dir, 'attack'))
sys.path.append(os.path.join(src_dir, 'models'))
sys.path.insert(0, './../../../Project1')
from utils import mkdir_p, AverageMeter, accuracy, print_acc_conf

# NOTE: Here is the victim model definition.
sys.path.insert(0, './../../models')
from purchase import PurchaseClassifier


# This is our adversary model we import from within the models directory
from adversary import AttackModel


# Privacy engine
privacy_engine = PrivacyEngine()

# Set to run on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def train_model(model, train_data, train_label, epochs, optimizer, batch_size, differentially_private_on=False):
    model = model.to(device, torch.float)
    criterion = nn.CrossEntropyLoss().to(device, torch.float)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_data, dtype=torch.float32),
        torch.tensor(train_label, dtype=torch.long))


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Change parameters to acheive best DP and calculate a predicted noise multiplier for a given epsilon
    if differentially_private_on:

        epsilon = float("inf")
        delta = 1e-5
        optimal_nm = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta, sample_rate=batch_size/train_data.shape[0], epochs=epochs)
        print(f"Calculated noise multiplier for epsilon {epsilon} : {optimal_nm}")

        model, optimizer, train_loader = privacy_engine.make_private(

            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=optimal_nm,
            max_grad_norm=1.0,

            )
        
        

    saved_epoch = 0
    best_acc = 0.0

    for epoch in range(epochs):

        running_loss = 0.0

        for _batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
        
            # Forward pass
            outputs = model(inputs)
            
            # Compute the loss
            loss = criterion(outputs, labels)
        
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track the loss
            running_loss += loss.item()
        
        # Print the average loss for the epoch 
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        if differentially_private_on:
            print(f"Epsilon: {privacy_engine.get_epsilon(delta)}")
        



def test_model(model, test_data, test_label, batch_size):

    model.eval()
    criterion = nn.CrossEntropyLoss()

    test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)

    losses = AverageMeter()
    top1 = AverageMeter()

    len_t = int(np.ceil(len(test_data)/batch_size))
    infer_np = np.zeros((len(test_data), 100))

    for batch_ind in range(len_t):
        # measure data loading time
        end_idx = min(len(test_data), (batch_ind+1)*batch_size)
        inputs = test_data_tensor[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets = test_label_tensor[batch_ind*batch_size: end_idx].to(device, torch.long)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets) 

        # measure accuracy and record loss
        prec1, prec2 = accuracy(outputs.data, targets.data, topk=(1,2))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])

    print('Average Testing Accuracy: {:.4f}.'.format(top1.avg))
    sys.stdout.flush()


# This function builds an attack dataset for training the attack model.
# Given a model, train_data, and test_data, this function will compute
# the softmax outer layer and append it to each input sample for inclusion
# in the attack dataset, with label 0 if the sample comes from test_data,
# and 1 if the sample comes from train_data.
def attack_data(model, train_data, train_label, test_data, test_label):

    model.eval()
    
    train_inputs = torch.from_numpy(train_data).type(torch.FloatTensor).to(device)
    train_outputs = F.softmax(model(train_inputs),dim=1)

    test_inputs = torch.from_numpy(test_data).type(torch.FloatTensor).to(device)
    test_outputs = F.softmax(model(test_inputs),dim=1)  

    zerovec = np.full(len(test_data), 0)
    onevec = np.full(len(train_data), 1)


    pre_xta1 = torch.cat((train_outputs, test_outputs)).cpu().detach().numpy()
    pre_xta2 = np.reshape(np.vstack((train_label, test_label)), (2 * train_label.size))
    pre_xta2_onehot = (F.one_hot(torch.from_numpy(pre_xta2).type(torch.LongTensor), num_classes=100)).cpu().numpy()
    data_a = np.hstack((pre_xta1, pre_xta2_onehot))

    
    label_a = np.hstack((onevec,zerovec))

    return data_a, label_a


# Use this to format our results when manually evaluating model
def format_result(results):
    no = 0
    yes = 0
    for i, row in enumerate(results):
        if row[0] > row[1]:
            no += 1
        else:
            yes +=1 
    print(f"Yes: {yes} | No: {no} ")
            
    
def main():
    parser = argparse.ArgumentParser(description='undefend training for Purchase dataset')
    parser.add_argument('--attack_epochs', type = int, default = 150, help = 'attack epochs in NN attack')
    parser.add_argument('--classifier_epochs', type = int, default = 30, help = 'classifier epochs')
    parser.add_argument('--batch_size', type = int, default = 512, help = 'batch size')
    parser.add_argument('--num_class', type = int, default = 100, help = 'num class')
    parser.add_argument('--lr', type = float, default = .001, help = 'learning rate')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    batch_size = args.batch_size
    num_class = args.num_class
    attack_epochs = args.attack_epochs
    classifier_epochs = args.classifier_epochs
    lr = args.lr

    DATASET_PATH = os.path.join(root_dir, 'purchase', 'data')

    
    
    # ============ VICTIM CLASSIFIER ============
    # Training the victim model. Hyperparameters can be provided at command-line
    # with defaults defined at beginning of main above. Note that the victim model
    # neural architecture is defined in MIAdefenseSELENA/models/purchase.py.

    enable_differential_privacy = True

    train_data_v = np.load(os.path.join(DATASET_PATH, 'partition', 'train_data_v.npy'))
    train_label_v = np.load(os.path.join(DATASET_PATH, 'partition', 'train_label_v.npy'))
    test_data_v = np.load(os.path.join(DATASET_PATH, 'partition', 'test_data_v.npy'))
    test_label_v = np.load(os.path.join(DATASET_PATH, 'partition', 'test_label_v.npy'))


    print("VICTIM CLASSIFIER TRAINING/EVALUATION")
    print(f"Differential Privacy status: {enable_differential_privacy}")
    
    model_v = PurchaseClassifier()
    optimizer = optim.Adam(model_v.parameters(), lr=lr)

    train_model(model_v, train_data_v, train_label_v, classifier_epochs, optimizer, batch_size, enable_differential_privacy)
    

    # ============ SHADOW MODEL ============
    # Create Shadow model based off of the Victim Classifier and then train with generated datasets

    train_data_s = np.load(os.path.join(DATASET_PATH, 'partition', 'train_data_s.npy'))
    train_label_s = np.load(os.path.join(DATASET_PATH, 'partition', 'train_label_s.npy'))
    test_data_s = np.load(os.path.join(DATASET_PATH, 'partition', 'test_data_s.npy'))
    test_label_s = np.load(os.path.join(DATASET_PATH, 'partition', 'test_label_s.npy'))

    print("SHADOW CLASSIFIER TRAINING/EVALUATION")

    model_s = PurchaseClassifier()
    optimizer = optim.Adam(model_s.parameters(), lr=lr)

    train_model(model_s, train_data_s, train_label_s, classifier_epochs, optimizer, batch_size)

    

    # ============ ATTACK MODEL ============

    train_data_a, train_label_a = attack_data(model_s, train_data_s, train_label_s, test_data_s, test_label_s)
    test_data_a, test_label_a = attack_data(model_v, train_data_v, train_label_v, test_data_v, test_label_v)

    print("ATTACK CLASSIFIER TRAINING/EVALUATION")
    
    model_a = AttackModel()
    optimizer = optim.Adam(model_a.parameters(), lr=lr)

    train_model(model_a, train_data_a, train_label_a, classifier_epochs, optimizer, batch_size)



    torch.set_printoptions(precision=2)

    # ============ TEST CLASSIFIERS ============
    
    print("Victim Model Testing:")
    test_model(model_v, test_data_v, test_label_v, batch_size)

    print("Shadow Model Testing:")
    test_model(model_s, test_data_s, test_label_s, batch_size)

    print("Attack Model Testing:")
    test_model(model_a, test_data_a, test_label_a, batch_size)


if __name__ == '__main__':
    main()