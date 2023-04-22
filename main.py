import torch
import torch.optim as optim
from compute_err_share_network import create_batch,comp_recall_userbased,comp_recall_userbased_4attention
from make_model import  *
import math
import time
import numpy as np
import pickle
import random
import collections
from datetime import datetime


import os
import time
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
parser = argparse.ArgumentParser(description='Shared representation cold-start recommendation')

#fpath: path to the dataset
parser.add_argument('--ds', default='citeulike', type=str)

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--model', default='flexible_model', type=str, help='model name')
parser.add_argument('--attention_type', default='cosin', type=str) # cosin, dot, or general
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--epoch', default=50, type=int, help='number of epoch')

#remove_item: Removes item from the users' support set.
parser.add_argument('--remove_item', dest='remove_item', action='store_true')

#MLP_user: Add MLP on top of the user representation
parser.add_argument('--MLP_user', dest='MLP_user', action='store_true')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
fpath = args.ds #path to the data

# load data, see README file for more details.
trainset = np.load('./' + args.ds + '/trainset.npy')

pkl_file = open('./'+fpath+'/item_dict.pkl', 'rb')
IFeature = pickle.load(pkl_file)
pkl_file.close()

interaction_matrix = np.load('./' + args.ds + '/interaction_matrix.npy')

pkl_file = open('./'+fpath+'/user_dict.pkl', 'rb')
uFeature = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('./'+fpath+'/val_grtr.pkl', 'rb')
val_grtr_userbased = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('./'+fpath+'/test_item_ids.pkl', "rb")
test_item_ids = pickle.load(pkl_file)
pkl_file.close()

train_size = len(trainset)
batch_size = args.batch_size
num_epochs=args.epoch

print("Create ", args.model)

if args.model == "flexible_model":
    model = flexible_model( item_feature = len(IFeature[1]), interaction_matrix=len(interaction_matrix[1]), concate_matrix=100, attention_type = args.attention_type, MLP_user  = args.MLP_user)

if args.model == "flexible_model":
    attn = True
else:
    attn = False


if device == 'cuda':
    print("Use gpu")
    model = model.cuda()


#Loss function and optimizer
criterion = innerproduct()
optimizer = optim.SGD(model.parameters(),lr = args.lr , momentum=0.9, weight_decay=1e-6)

# num_iters is the number of iterations per epoch
num_iters = math.ceil(len(trainset) / batch_size)

recall = comp_recall_userbased_4attention(model, IFeature, uFeature,interaction_matrix,
                        val_grtr_userbased, K=100, device= device, remove_item = args.remove_item,attn=attn)

print("initial recall:", recall)

# store recalls
recall = 0
best_recall = 0


# This is the main loop
for epoch in range(0, num_epochs):
    start=time.time()
    #shuffle data at the beginning of each epoch
    shuffle_indices = np.random.permutation(train_size)
    shuffled_data = trainset[shuffle_indices,:]

    # shuffle data at the beginning of each epoch
    shuffle_indices = np.random.permutation(train_size)
    shuffled_data = trainset[shuffle_indices,:]

    val_user_keys = val_grtr_userbased.keys()
    val_user_ids = list(val_user_keys)

    for batch_num in range(num_iters):
        # extract training data for a batch of size "batch_size"
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, train_size)
        data_train = shuffled_data[start_index:end_index,:]
        
        label = data_train[:, 2].reshape(-1, 1)
        
        if device == 'cuda':
            label = torch.tensor(label).cuda().long()
        else:
            label = torch.LongTensor(label)

        np_uid = data_train[:, 0]#.reshape(-1, 1).astype(int)
        np_iid = data_train[:, 1].reshape(-1, 1).astype(int)

        # re-weight in batch
        label_view=label.view(-1)
        label_list=label_view.tolist()
        container=collections.Counter(label_list)

        count_pos=container.get(1)
        count_neg=container.get(0)
        if count_pos==None:
            count_pos=0
        if count_neg==None:
            count_neg=0

        weight=[]
        
        for i in range(len(label)):
            if label[i][0]==1:
                B=(1-pow(0.99,count_pos))/(1-0.99)
                sample_positive=1/B
                weight.append(sample_positive)
            elif label[i][0]==0:
                B=(1-pow(0.99,count_neg))/(1-0.99)
                sample_negative=1/B
                weight.append(sample_negative)
        weight=torch.tensor(weight).to(device)
        
        # Create a batch of the data
        inp_u_index, inp_u_vocabs, inp_i_vocab, intex_mat, interaction_matrix_batch = create_batch(np_uid, np_iid, IFeature, 
                                                                                        uFeature, interaction_matrix, attn=attn,user_data=True, remove_item=args.remove_item)
        optimizer.zero_grad()

        if device == 'cuda':
            inp_u_index, inp_u_vocabs,inp_i_vocab,intex_mat, interaction_matrix_batch, IFeature2 = torch.from_numpy(inp_u_index).cuda().float(), torch.from_numpy(inp_u_vocabs).cuda().float(), torch.from_numpy(inp_i_vocab).cuda().float(), torch.from_numpy(intex_mat).cuda().float(), torch.from_numpy(interaction_matrix_batch).cuda().float(), torch.from_numpy(IFeature).cuda().float()
        else:    
            inp_u_index, inp_u_vocabs,inp_i_vocab, intex_mat= torch.from_numpy(inp_u_index).float(), torch.from_numpy(inp_u_vocabs).float(), torch.from_numpy(inp_i_vocab).float(), torch.from_numpy(intex_mat).float()
        
        # Train the model using this batch
        output1,output2, output3_appendusertower = model(inp_u_vocabs,inp_i_vocab,intex_mat, interaction_matrix_batch)
        loss_contrastive = criterion(output1,output2,output3_appendusertower, label,weight, device)
        loss_contrastive.backward()
        clipping_value = 1 # arbitrary value of your choosing
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
        optimizer.step()
        
    if args.model == "flexible_model":
        recall = comp_recall_userbased_4attention(model, IFeature, uFeature, interaction_matrix,
                           val_grtr_userbased, K=100, device = device, remove_item = args.remove_item,attn=attn)
    print( 'test recall:', recall, 'Epoch ', epoch+1, 'now: ', datetime.now())
    


    if recall > best_recall:
        best_recall = recall

print('Best test recall:', best_recall)