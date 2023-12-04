"""
Reference:
https://github.com/FedML-AI/FedML
"""
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from scipy.optimize import root
import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
import torch
import numpy as np

import numpy as np
from scipy.optimize import root,bisect
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch as t
from torch.nn import functional as F





def calc_ent(_x):
    """
        calculate shanno ent of x
    """
    ent = 0.0
    for x_value in _x:
        p = float(x_value)
        logp = np.log2(p)
        ent -= p * logp
    return ent

class SoftenerSearchOperator:
    def __init__(self,vec,target_ent):
        self.vec=vec
        self.target_ent=target_ent
        
    def __call__(self,S):
        #注意，这个softener是放到softmax之前的，最好与外层的DKC联合起来
        softened_ent = calc_ent(F.softmax(self.vec / S) + 10 ** (-7))
        return softened_ent-self.target_ent


class DistributedKnowledgeCongruence_search_based(nn.Module):
    def __init__(self,target_ent,C):
        super(DistributedKnowledgeCongruence_search_based, self).__init__()
        self.target_ent=target_ent
        self.C=C
    def forward(self,logits):#C表示类别数
        all_new_knowledge=[]
        for idx in range(logits.shape[0]): 
            all_new_knowledge.append(
                np.array(self.generate_congruent_knowledge(logits[idx]).detach().cpu())
                )
        return torch.Tensor(np.array(all_new_knowledge)).cuda()

    def generate_congruent_knowledge(self,x):
        search_operator=SoftenerSearchOperator(x,self.target_ent)
        result_softener=bisect(search_operator, 1e-4,1000)
        return F.softmax(x / result_softener)

        
class DKC_KL_Loss_search_based(nn.Module):
    def __init__(self,temperature,target_ent,C):
        self.T=temperature
        super(DKC_KL_Loss_search_based, self).__init__()
        self.DKC=DistributedKnowledgeCongruence_search_based(target_ent,C)

    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # teacher_outputs -> B X num_classes

        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)
        output_batch,teacher_outputs=output_batch.cuda(),teacher_outputs.cuda()

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        #teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)
        teacher_outputs = self.DKC(teacher_outputs)
        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        # Same result KL-loss implementation
        # loss = T * T * torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
        return loss.cuda()


class DistributedKnowledgeCongruence(nn.Module):
    def __init__(self,T,S,C):
        super(DistributedKnowledgeCongruence, self).__init__()
        self.T=T
        self.S=S
        self.C=C
    def forward(self,logits):#C表示类别数
        all_new_knowledge=[]
        for idx in range(logits.shape[0]): 
            all_new_knowledge.append(
                np.array(self.generate_congruent_knowledge(logits[idx]).detach().cpu())
                )
        return torch.Tensor(np.array(all_new_knowledge)).cuda()

    def generate_congruent_knowledge(self,x):
        n=x.shape[0]
        #print("old knowledge:",x)
        max_num=float(t.max(x))
        #print("x:",x)
        #print("up:",(self.C*self.T-1)*x+max_num-self.T)
        new_knowledge=((self.C*self.T-1)*x+max_num-self.T)/(self.C*max_num-1)
        #print("new knowledge:",new_knowledge)
        min_new_knowledge=float(t.min(new_knowledge))
        #print("old knowledge:",x)
        #print("new knowledge:",new_knowledge)
        if min_new_knowledge<0:
            #print("rectify the knowledge")
            #print("min_new_knowledge<0")
            max_ind=t.argmax(x)
            #print("max_ind:",max_ind)
            new_knowledge=torch.full_like(new_knowledge,(1-self.T)/(self.C-1))
            new_knowledge[max_ind]=self.T
        else:
            pass
            #print("not rectify the knowledge")
            #print("changed new knowledge:",new_knowledge)
        #print("sum_new_knowledge:",torch.sum(new_knowledge))
        #print("final knowledge:",new_knowledge)
        return new_knowledge



def get_state_dict(file):
    try:
        pretrain_state_dict = torch.load(file)
    except AssertionError:
        pretrain_state_dict = torch.load(file, map_location=lambda storage, location: storage)
    return pretrain_state_dict


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        #print("total",self.total)
        #print("steps",self.steps)
        self.total = self.total+val
        self.steps = self.steps+1

    def value(self):
        return self.total / float(self.steps)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        #correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



class DKC_KL_Loss(nn.Module):
    def __init__(self,temperature,T,S,C):
        self.T=temperature
        super(DKC_KL_Loss, self).__init__()
        self.DKC=DistributedKnowledgeCongruence(T,S,C)
        self.S=S

    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # teacher_outputs -> B X num_classes

        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)
        output_batch,teacher_outputs=output_batch.cuda(),teacher_outputs.cuda()

        teacher_outputs=teacher_outputs/self.S

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)
        teacher_outputs = self.DKC(teacher_outputs)
        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        # Same result KL-loss implementation
        # loss = T * T * torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
        return loss.cuda()




class KL_Loss(nn.Module):
    def __init__(self, temperature=3.0):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):

        # output_batch  -> B X num_classes
        # teacher_outputs -> B X num_classes

        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        # Same result KL-loss implementation
        # loss = T * T * torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
        return loss


class CE_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(CE_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch      -> B X num_classes
        # teacher_outputs   -> B X num_classes

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1)

        # Same result CE-loss implementation torch.sum -> sum of all element
        loss = -self.T * self.T * torch.sum(torch.mul(output_batch, teacher_outputs)) / teacher_outputs.size(0)

        return loss


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


# Filter out batch norm parameters and remove them from weight decay - gets us higher accuracy 93.2 -> 93.48
# https://arxiv.org/pdf/1807.11205.pdf
def bnwd_optim_params(model, model_params, master_params):
    bn_params, remaining_params = split_bn_params(model, model_params, master_params)
    return [{'params': bn_params, 'weight_decay': 0}, {'params': remaining_params}]


def split_bn_params(model, model_params, master_params):
    def get_bn_params(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): return module.parameters()
        accum = set()
        for child in module.children(): [accum.add(p) for p in get_bn_params(child)]
        return accum

    mod_bn_params = get_bn_params(model)
    zipped_params = list(zip(model_params, master_params))

    mas_bn_params = [p_mast for p_mod, p_mast in zipped_params if p_mod in mod_bn_params]
    mas_rem_params = [p_mast for p_mod, p_mast in zipped_params if p_mod not in mod_bn_params]
    return mas_bn_params, mas_rem_params



class KL_Loss_equivalent(nn.Module):
    def __init__(self, temperature=1):    
        super(KL_Loss_equivalent, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = self.T * self.T * \
                    torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
        return loss


class FPKD(nn.Module):
    def __init__(self, temperature=1,dist_local=None,T=None):    
        super(FPKD, self).__init__()
        self.T = temperature
        self.dist_local=dist_local
        self.FPKD_T=T
        self.KL=KL_Loss()

    def forward(self, output_batch, teacher_outputs):

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = self.T * self.T * \
                    torch.sum(torch.sum(
                        torch.mul((self.dist_local/self.FPKD_T).cuda(),
                        torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))
                              )/teacher_outputs.size(0)
        return loss+self.KL(output_batch,teacher_outputs)


class LKA_balance(nn.Module):
    def __init__(self, temperature=1,dist_global=None,dist_local=None,U=None):    
        super(LKA_balance, self).__init__()
        self.T = temperature
        self.dist_global=dist_global
        self.dist_local=dist_local
        self.U=U
        self.KL=KL_Loss()

    def forward(self, output_batch, teacher_outputs):
        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = self.T * self.T * \
                    torch.sum(torch.sum(
                        torch.mul((self.dist_global-self.dist_local).cuda()/self.U,
                        torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))
                              )/teacher_outputs.size(0)
        return loss+self.KL(output_batch,teacher_outputs)

class LKA_sim(nn.Module):
    def __init__(self, temperature=1,dist_global=None,dist_local=None):    
        super(LKA_sim, self).__init__()
        self.T = temperature
        self.dist_global=dist_global
        self.dist_local=dist_local
        self.KL=KL_Loss()
    def forward(self, output_batch, teacher_outputs):

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = self.T * self.T * \
                    torch.sum(t.cosine_similarity(self.dist_global,self.dist_local,dim=0)*
                        torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch))
                        )/teacher_outputs.size(0)
        return loss+self.KL(output_batch,teacher_outputs)

        
