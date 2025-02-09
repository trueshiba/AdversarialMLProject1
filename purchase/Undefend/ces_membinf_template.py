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
from torchsummary import summary

config_file = './../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']

sys.path.append(src_dir)
sys.path.append(root_dir)
# print(root_dir)
# print(src_dir)
sys.path.append(os.path.join(src_dir, 'attack'))
sys.path.append(os.path.join(src_dir, 'models'))
sys.path.insert(0, './../../../Project1')
from utils import mkdir_p, AverageMeter, accuracy, print_acc_conf

# NOTE: Here is the victim model definition.
sys.path.insert(0, './../../models')
from purchase import PurchaseClassifier
from adversary import AttackModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def train(train_data, train_labels, model, criterion, optimizer, batch_size):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    len_t = int(np.ceil(len(train_data)/batch_size))

    for batch_ind in range(len_t):
        end_idx = min((batch_ind+1)*batch_size, len(train_data))
        inputs = train_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets = train_labels[batch_ind*batch_size: end_idx].to(device, torch.long)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec2 = accuracy(outputs.data, targets.data, topk=(1, 2))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])

    return (losses.avg, top1.avg)

def train_eval(test_data, labels, model, criterion, batch_size):
    # switch to evaluate mode
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    len_t = int(np.ceil(len(test_data)/batch_size))
    infer_np = np.zeros((len(test_data), 100))

    for batch_ind in range(len_t):
        # measure data loading time
        end_idx = min(len(test_data), (batch_ind+1)*batch_size)
        inputs = test_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets = labels[batch_ind*batch_size: end_idx].to(device, torch.long)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # DISABLED conf calculation due to conflict with attack training
        #infer_np[batch_ind*batch_size: end_idx] = (F.softmax(outputs,dim=1)).detach().cpu().numpy()
        #conf = np.mean(np.max(infer_np[batch_ind*batch_size:end_idx], axis = 1))
        #confs.update(conf, inputs.size()[0])        

        # measure accuracy and record loss
        prec1, prec2 = accuracy(outputs.data, targets.data, topk=(1,2))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])

    return (losses.avg, top1.avg, 0)


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def train_model(model, train_data, train_label, test_data, test_label, epochs, batch_size):
    model = model.to(device,torch.float)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device, torch.float)

    train_data_tensor = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)

    test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)

    saved_epoch = 0
    best_acc = 0.0

    for epoch in range(1, epochs+1):
        r= np.arange(len(train_data))
        np.random.shuffle(r)
        train_data = train_data[r]
        train_label = train_label[r]
        train_data_tensor = torch.from_numpy(train_data).type(torch.FloatTensor)
        train_label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)

        train_loss, trainning_acc = train(train_data_tensor, train_label_tensor, model, criterion, optimizer, batch_size)

        train_loss, train_acc, train_conf = train_eval(train_data_tensor, train_label_tensor, model, criterion, batch_size)
        
        test_loss, test_acc, test_conf = train_eval(test_data_tensor,test_label_tensor, model, criterion, batch_size)

        # save model
        is_best = test_acc>best_acc
        best_acc = max(test_acc, best_acc)

        if is_best:
            saved_epoch = epoch

        print('Epoch: [{:d} | {:d}]: training loss: {:.4f} training|train|test acc: {:.4f}|{:.4f}|{:.4f}.'.format(epoch, epochs, train_loss, trainning_acc, train_acc, test_acc))
        sys.stdout.flush()

# This function builds an attack dataset for training the attack model.
# Given a model, train_data, and test_data, this function will compute
# the softmax outer layer and append it to each input sample for inclusion
# in the attack dataset, with label 0 if the sample comes from test_data,
# and 1 if the sample comes from train_data.
def attack_data(model, train_data, test_data):

    # if not train_data.is_cuda:
    #     train_data.to(device)
    # elif not test_data.is_cuda:
    #     test_data.to(device)

    model.eval()
    
    train_inputs = torch.from_numpy(train_data).type(torch.FloatTensor).to(device)
    train_outputs = F.softmax(model(train_inputs),dim=1)
    
    test_inputs = torch.from_numpy(test_data).type(torch.FloatTensor).to(device)
    test_outputs = F.softmax(model(train_inputs),dim=1)  

    zerovec = np.full(len(test_data), 0)
    onevec = np.full(len(train_data), 1)

    pre_xta = torch.cat((train_outputs, test_outputs)).detach().cpu().numpy()
    data_a = np.hstack((np.vstack((train_inputs.detach().cpu().numpy(),test_inputs.detach().cpu().numpy())),
                        pre_xta))
    label_a = np.hstack((onevec,zerovec))

    return data_a, label_a


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

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    batch_size = args.batch_size
    num_class = args.num_class
    attack_epochs = args.attack_epochs
    classifier_epochs = args.classifier_epochs

    DATASET_PATH = os.path.join(root_dir, 'purchase',  'data')
  
    # Reading in the victim train and test sets, created with
    # MIAdefenseSELENA/purchase/ces_partition.py.
    
    train_data_v = np.load(os.path.join(DATASET_PATH, 'partition', 'train_data_v.npy'))
    train_label_v = np.load(os.path.join(DATASET_PATH, 'partition', 'train_label_v.npy'))
    

    dumby_data = torch.from_numpy(train_data_v[:10][:]).type(torch.FloatTensor)
    train_data_v = train_data_v[10:][:]
    train_label_v = train_label_v[10:][:]

    test_data_v = np.load(os.path.join(DATASET_PATH, 'partition', 'test_data_v.npy'))
    test_label_v = np.load(os.path.join(DATASET_PATH, 'partition', 'test_label_v.npy'))

    test_data_v = test_data_v[10:][:]
    test_label_v = test_label_v[10:][:]
    


    # Training the victim model. Hyperparameters can be provided at command-line
    # with defaults defined at beginning of main above. Note that the victim model
    # neural architecture is defined in MIAdefenseSELENA/models/purchase.py.
    
    print("VICTIM CLASSIFIER TRAINING/EVALUATION")
    model_v = PurchaseClassifier()
    train_model(model_v,
                train_data_v, train_label_v, test_data_v, test_label_v,
                classifier_epochs, batch_size)
    


    # Create Shadow model based off of the Victim Classifier and then train with generated datasets
    train_data_s = np.load(os.path.join(DATASET_PATH, 'partition', 'train_data_s.npy'))
    train_label_s = np.load(os.path.join(DATASET_PATH, 'partition', 'train_label_s.npy'))
    test_data_s = np.load(os.path.join(DATASET_PATH, 'partition', 'test_data_s.npy'))
    test_label_s = np.load(os.path.join(DATASET_PATH, 'partition', 'test_label_s.npy'))

    print("SHADOW CLASSIFIER TRAINING/EVALUATION")
    model_s = PurchaseClassifier()
    #summary(model_s, (512, 600))
    train_model(model_s,
                train_data_s, train_label_s, test_data_s, test_label_s,
                classifier_epochs, batch_size)

    

    # Train attack model on attack dataset
    print("ATTACK CLASSIFIER TRAINING/EVALUATION")
    model_a = AttackModel()

    # Use attack_data() to get attack model dataset
    data_a, label_a = attack_data(model_s, train_data_s, test_data_s)

    summary(model_a, (512, 700))
    # big_data = np.concatenate((train_data_v, test_data_v))
    # big_labels = np.concatenate((train_label_v, test_label_v))

    test_a, test_labels_a = attack_data(model_v, train_data_v, test_data_v)

    train_model(model_a, data_a, label_a, test_a, test_labels_a, classifier_epochs, batch_size)



    # Testing again on our random choice data
    model_a.eval()    
    torch.set_printoptions(precision=2)

    print("\nShould be all yes")  
    gen = np.random.default_rng()    
    randomSamples = torch.from_numpy(gen.choice(train_data_v, 10, replace=False)).type(torch.FloatTensor).to(device)    
    logits = model_a(torch.hstack((randomSamples.to(device), model_v(randomSamples).to(device))))
    format_result(F.softmax(logits, dim=1)*100)    

    print("\nShould be no")
    nonsenseSamples = torch.rand(randomSamples.shape)    
    logits = model_a(torch.hstack((nonsenseSamples.to(device), model_v(nonsenseSamples.to(device)).to(device))))    
    format_result(F.softmax(logits, dim=1)*100) 

    print("\nShould definitely be no")
    logits = model_a(torch.hstack((dumby_data.to(device), model_v(dumby_data.to(device)).to(device))))    
    dummy_perc = F.softmax(logits, dim=1)*100
    format_result(dummy_perc)
    






if __name__ == '__main__':
    main()