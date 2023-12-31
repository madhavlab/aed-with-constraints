import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import librosa
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from librosa import feature
import torch.nn as nn
from torchsummary import summary
from tqdm import tqdm
import torchvision.transforms as vtransforms
import dcase_util
from sed_eval.audio_tag import AudioTaggingMetrics as get_metrics_sedeval
from collections import OrderedDict
from torchlibrosa.augmentation import SpecAugmentation
import torch.nn.functional as Ft


from implication_code import Implication


class Constraints(nn.Module):
    
    def __init__(self,ontology={'parent':['child1','...childn']}):
        super(Constraints, self).__init__()
        self.ontology = ontology
        self.constraints = Implication(ontology = self.ontology)
        self.lambda_param = nn.Parameter(torch.zeros(self.constraints.num_constraints()))
    
    def get_optim_params(self,ddlr=0.06,use_wt_as_lr_factor = True):
        params = {'params':None,'lr':None}
        factor = 1
        if use_wt_as_lr_factor:
            factor = self.constraints.weight
            self.constraint_dict[k].weight = 1
        if self.lambda_param.requires_grad:
            params['params'] = self.lambda_param
            params['lr'] = ddlr*factor
        return params
        
    def forward(self,scores):

        h_k = self.constraints.get_penalty(scores)
        penalty = (self.lambda_param*h_k).sum() 
        loss = self.constraints.weight*penalty
        return loss,h_k