from syslog import LOG_SYSLOG
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Inner product of the user and item representations is used to predict the true label
class innerproduct(torch.nn.Module):
    def __init__(self):
        super(innerproduct, self).__init__()
    def forward(self, output1, output2, f_concate_rep, label, weight_, device):
        inner_product_withtower1 = torch.sum( torch.mul(output1, output2), 1).float()
        
        # sigmoid
        sigmoid1=nn.functional.sigmoid(inner_product_withtower1).clone()
        sigmoid2=nn.functional.sigmoid(f_concate_rep).clone()
        sigmoid=sigmoid1

        # gumbel softmax
        t1=torch.stack([inner_product_withtower1, f_concate_rep], dim=1)
        gumbel_result=torch.nn.functional.gumbel_softmax(t1, tau=0.7, hard=True).to(device)

        sigmoid=gumbel_result[:,0]*sigmoid1 + gumbel_result[:,1]*sigmoid2

        cri = torch.nn.BCELoss(weight=weight_, size_average=None, reduce=None, reduction='sum')
        loss_BCE = cri(sigmoid, label[:,0].float())

        return loss_BCE
 
class flexible_model(nn.Module):

    def __init__(self, item_feature, interaction_matrix, concate_matrix, attention_type = 'dot', MLP_user  = False):
        super(flexible_model, self).__init__()
        
        self.MLP_user = MLP_user

        # Layers for the attention module
        self.attention_type = attention_type
        
        #citeulike
        #Three fully connected layer for the item input
        self.fully_connected_1 = nn.Sequential(torch.nn.Linear(item_feature, 250), nn.Tanh())
        self.fully_connected_2 = nn.Sequential(torch.nn.Linear(250, 200),nn.Tanh())
        self.fully_connected_3 = nn.Sequential(torch.nn.Linear(200, 100),nn.Tanh())
        
        self.fully_connected_7 = nn.Sequential(torch.nn.Linear(interaction_matrix, 1000), nn.ReLU()) 
        self.fully_connected_8 = nn.Sequential(torch.nn.Linear(1000, 500),nn.ReLU())
        self.fully_connected_9 = nn.Sequential(torch.nn.Linear(500, 100),nn.ReLU())

        # # mlimdb
        # self.fully_connected_1 = nn.Sequential(torch.nn.Linear(item_feature, 1000), nn.Tanh())
        # self.fully_connected_2 = nn.Sequential(torch.nn.Linear(1000, 500),nn.Tanh())
        # self.fully_connected_3 = nn.Sequential(torch.nn.Linear(500, 100),nn.Tanh())

        # self.fully_connected_7 = nn.Sequential(torch.nn.Linear(interaction_matrix, 1000), nn.ReLU()) 
        # self.fully_connected_8 = nn.Sequential(torch.nn.Linear(1000, 500),nn.ReLU())
        # self.fully_connected_9 = nn.Sequential(torch.nn.Linear(500, 100),nn.ReLU())
        
        
        self.fully_connected_10 = nn.Sequential(torch.nn.Linear(concate_matrix, 80),nn.ReLU())
        self.fully_connected_11 = nn.Sequential(torch.nn.Linear(80, 60),nn.ReLU())
        self.fully_connected_12 = nn.Sequential(torch.nn.Linear(60, 40),nn.ReLU())
        

        #if self.MLP_user: Two fully connected layer on top of the user representation
        self.fully_connected_4 = nn.Sequential(torch.nn.Linear(100, 100),nn.Tanh())
        self.fully_connected_5 = nn.Sequential(torch.nn.Linear(100, 100),nn.Tanh())
        
        
    '''
    Makes items representation

    aregs:
        inp_vocab: the items' feature vectors
    '''
    def forward_once(self, inp_vocab):
        f_irep = self.fully_connected_1(inp_vocab)
        f_irep = self.fully_connected_2(f_irep)
        f_irep = self.fully_connected_3(f_irep)
        return f_irep

    def forward_once2(self, inp_vocab):
        f_irep = self.fully_connected_7(inp_vocab)
        f_irep = self.fully_connected_8(f_irep)
        f_irep = self.fully_connected_9(f_irep)
        return f_irep

    def forward_once3(self, inp_vocab):
        f_irep = self.fully_connected_10(inp_vocab)
        f_irep = self.fully_connected_11(f_irep)
        f_irep = self.fully_connected_12(f_irep)
        return f_irep
    

    '''
    Returns both item and user representations

    args:
        u_inp_vocabs: index of the items in the users' support set for each user in the batch
        i_inp_vocab: item features vectors of the items in the batch
        intex_mat: a matrix to combine item representations of the items each user interacted with

    '''

    def forward(self, u_inp_vocabs, i_inp_vocab, intex_mat, interaction_matrix_batch):
        f_urep = self.forward_once(u_inp_vocabs)
        f_irep = self.forward_once( i_inp_vocab)
        f_urep_ver2 = self.forward_once2(interaction_matrix_batch)
        concate = torch.mul(f_urep_ver2, f_irep)
        f_concate_rep = self.forward_once3(concate)
        f_concate_rep= torch.sum(f_concate_rep, 1).float()

        XX = torch.matmul(torch.transpose(intex_mat.clone(),0,1),f_irep)
        

        if(self.attention_type == 'cosin'):
            coss_loss = nn.CosineSimilarity(dim=1, eps=1e-6)
            Z = coss_loss(XX, f_urep)
        elif(self.attention_type == 'dot'):
            Z = torch.sum(torch.mul(XX, f_urep), 1)
        elif(self.attention_type == 'general'):
            u_a_rep = self.attention(f_urep)
            Z = torch.sum(torch.mul(XX, u_a_rep), 1)
            
            
        Z = torch.unsqueeze(Z,1)
        Z = torch.exp(Z)
        sumA = torch.matmul(intex_mat,Z)
        sumB = torch.matmul(torch.transpose(intex_mat.clone(),0,1),sumA)
        norm_Z = torch.div(Z,sumB)

        f_urep2 = norm_Z * f_urep  # multiply it by the user reps
        f_urep = torch.matmul(intex_mat, f_urep2)  # final user representations: weighted sum
    

        if self.MLP_user:
            f_urep = self.fully_connected_4(f_urep)
            f_urep = self.fully_connected_5(f_urep) 
            
        return f_urep, f_irep, f_concate_rep 
        

