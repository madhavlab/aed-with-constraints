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

class SED(nn.Module):
    def __init__(self,num_children=10,num_parents=4):
        super(SED,self).__init__()
        self.num_children = num_children
        self.num_parents = num_parents
        self.cnn1 = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU()]) #16, 128, 128, 321
        self.cnn2 = nn.Sequential(*[nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU()])#16, 256, 128, 321
        self.cnn3 = nn.Sequential(*[nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU()])
        # self.cnn4 = nn.Sequential(*[nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        #                             nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #                             nn.ReLU()])
        self.fc1 = nn.Linear(512, 320, bias=True)
        self.fc_final = nn.Sequential(*[nn.Linear(in_features=320, out_features=192, bias=True),
                                      nn.ReLU(),
                                      nn.Dropout(0.3),
                                      nn.Linear(in_features=192, out_features=int(num_children+num_parents), bias=True)])
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)    

    def forward(self,x):
        x = self.spec_augmenter(x)
        x = self.cnn1(x) #16, 128, 128, 321
        x = Ft.dropout(Ft.avg_pool2d(x,kernel_size=(2,2)),p=0.2) #16, 128, 64, 160
        x = self.cnn2(x) #16, 256, 64, 160
        x = Ft.dropout(Ft.avg_pool2d(x,kernel_size=(2,2)),p=0.2) #16, 256, 32, 80
        x = self.cnn3(x) #16, 1024, 32, 80
        x = Ft.avg_pool2d(x,kernel_size=(2,2)) #16, 1024, 16, 40
        # x = self.cnn4(x)
        x = Ft.dropout(x,p=0.2)
        x = torch.mean(x, dim=3) #16, 1024, 16
        (x1, _) = torch.max(x, dim=2) #16, 1024
        x2 = torch.mean(x, dim=2) #16, 1024
        x = x1 + x2 #16, 1024
        x = Ft.relu_(self.fc1(x)) #16, 1024 
        x = self.fc_final(x)
        return x
    
