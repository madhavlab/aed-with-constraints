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


class Implication:
    def __init__(self,ontology,weight = 1,device=torch.device('cuda')):
        self.ontology = ontology
        counting = lambda ontology:{key:len(value) for key,value in ontology.items()}
        self.counts = counting(self.ontology)
        self.cids = list(range(sum([num_children for num_children in self.counts.values()])))
        self.pids = list(range(self.cids[-1]+1,14))
        self.weight = weight
        self.device = device
    
    def num_constraints(self):
        return len(self.ontology.keys())
    
    def get_penalty(self,scores): # scores: shape is (batch_size,num_nodes)
        return self._get_h_over_batch(pred_proba_children = scores[:,self.cids],
                                      pred_proba_parents = scores[:,self.pids])
    
    def hinge(self,C):
        if C<0:
            return -C
        else:
            return 0
    
    def _get_H_per_example(self,
                           pred_proba_children,
                           pred_proba_parents): 
        #counts- number of children per parent. an array/tensor
        #pred_proba- tensor of probabilities
        # returns K constraints per example
        H_i = torch.zeros((len(self.ontology),)).to(self.device) #The K constraints for the i^th example
        start = 0
        k=0
        for parent,children in self.ontology.items():
            H_i[k] = self.hinge(pred_proba_parents[k]-sum(pred_proba_children[start:start+self.counts[parent]]))
            start += self.counts[parent]
            k+=1
        return H_i
    
    def _get_h_over_batch(self,
                          pred_proba_children,
                          pred_proba_parents):
        # returns K constraints over the batch
        h = torch.zeros((len(self.ontology),)).to(self.device)
        for i in range(pred_proba_children.shape[0]):
            h += self._get_H_per_example(pred_proba_children = pred_proba_children[i,:],
                                    pred_proba_parents = pred_proba_parents[i,:])
        return h