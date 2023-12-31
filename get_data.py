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



class getdata(Dataset):
    def __init__(self,
                 dataset_to_use = ['fsd50k','usound8k','msos'],
                 load_existing = False,
                 
                 window_dur = 2,
                 target_sr = 32000,
                 
                 do_augment = False,
                 label_size=5,
                 augmented_samples_per_ref=4,
                 
                 chunk_dur = 4,
                 chunk_thresh_dur=2,
                 
                 mode = 'train',
                 device=torch.device('cuda:0'),
                 save_path = 'processed_annotations',
                 
                 fold = 10):
        r"""
        Load the preprocessed audio files as a Pytorch dataset.
        
        Args:
            dataset_to_use (List, optional): which of the available datasets to be used. (Default: ``['fsd50k','usound8k','msos']``)
            load_existing (bool, optionsl): Whether to reload the mentioned datasets (False) or to process the mentioned datasets with given transformations before saving and loading. (Default ``False``)
                 
            window_dur (float, optional): The duration (in seconds) of the chunks of the audio files in the dataset. (Default: ``2.0``),
            target_sr (int, optional): The to be sample rate of the processed audio files. (Default: ``44100``),

            do_augment = False,
            label_size=5,
            augmented_samples_per_ref=4,

            device=torch.device('cuda:0')
        """
        self.dataset_to_use = [d.upper() for d in dataset_to_use]
        self.device = device
        self.target_sr = target_sr
        
        path_to_audio = 0
        path_to_gt=0
        self.data_path = '../../datasets/usound8k/'
        
        
        
        self.window = window_dur*self.target_sr
        
        self.chunk_length = int(chunk_dur*target_sr)
        self.chunk_thresh = int(target_sr*chunk_thresh_dur)
        self.mode = mode
        save_path = os.path.join(self.data_path,save_path)
        if not load_existing:
            if 'USOUND8K' in self.dataset_to_use:
            
                self.usound_parents= OrderedDict({'dog_bark':'living_things',
                                                  'children_playing':'living_things',
                                                  'air_conditioner':'mechanical',
                                                  'engine_idling':'mechanical',
                                                  'gun_shot':'mechanical',
                                                  'street_music':'street',
                                                  'siren':'street',
                                                  'car_horn':'street',
                                                  'jackhammer':'tools',
                                                  'drilling':'tools'})
                self.ontology = OrderedDict()
                self.counts = []
                
                for key,value in self.usound_parents.items():
                    if value in self.ontology.keys():
                        self.ontology[value].append(key)
                    else:
                        self.ontology[value] = [key]
                self.unique_labels = list(self.usound_parents.keys())
                parents = list(self.ontology.keys())
                for z in range(len(parents)):
                    self.unique_labels.append(parents[z])
                self.unique_labels = pd.DataFrame(self.unique_labels,columns = ['label2ids'])
                self.dataset = pd.read_csv(os.path.join(self.data_path,'UrbanSound8K/metadata/UrbanSound8K_meta.csv'))
                self.dataset.drop(np.where(self.dataset['length'] < 0.4*self.dataset['length'].mean())[0],
                                        inplace=True)
                self.dataset.reset_index(inplace=True,drop=True)
                self.dataset.drop(np.intersect1d(np.where(self.dataset['length']>window_dur)[0],
                                                       np.where(self.dataset['length']<1.5*window_dur)[0]),
                                        inplace=True)
                self.dataset.reset_index(inplace=True,drop=True)
                self.dataset = self.dataset.loc[:,['filepath','class']]
                self.dataset.rename(columns = {'class':'label1','filepath':'fpath'},inplace=True)
                self.dataset['label2'] = self.dataset['label1']
                self.dataset['label2'] = self.dataset['label2'].replace(self.usound_parents)
                self.dataset.reset_index(inplace=True,drop=True)   
                
            if do_augment:
                pass
            
            if not os.path.exists(save_path):
                os.mkdir(os.path.join(save_path))
                
            self.train = pd.DataFrame(np.zeros(self.dataset.shape),columns = self.dataset.columns)
            self.test = pd.DataFrame(np.zeros(self.dataset.shape),columns = self.dataset.columns)
            train_id,test_id = 0,0
            for idx in tqdm(range(self.dataset.shape[0])):
                if f'fold{fold}' in self.dataset['fpath'].iloc[idx]:
                    self.test.iloc[test_id,:] = self.dataset.iloc[idx,:]
                    test_id += 1
                else:
                    self.train.iloc[train_id,:] = self.dataset.iloc[idx,:]
                    train_id += 1
            self.train.drop(index = list(range(train_id,len(self.train))),inplace=True)
            self.test.drop(index = list(range(test_id,len(self.test))),inplace=True)
            
            self.dataset.to_csv(os.path.join(save_path,'processed_annotations_usound8k.csv'),index_label=0)
            self.train.to_csv(os.path.join(save_path,'processed_annotations_usound8k_train.csv'),index_label=0)
            self.test.to_csv(os.path.join(save_path,'processed_annotations_usound8k_test.csv'),index_label=0)
   
        else:
            self.dataset = pd.read_csv(os.path.join(save_path,'processed_annotations_usound8k.csv'),index_col=0)
            self.train = pd.read_csv(os.path.join(save_path,'processed_annotations_usound8k_train.csv'),index_col=0)
            self.test = pd.read_csv(os.path.join(save_path,'processed_annotations_usound8k_test.csv'),index_col=0)
            
        labels = pd.concat((pd.DataFrame(self.dataset['label1'].unique(),columns = ['Labels']),
                            pd.DataFrame(self.dataset['label2'].unique(),columns = ['Labels'])))
        self.events2labels = {labels.iloc[i,0]:torch.Tensor([i]).long() for i in range(len(labels))} 
        self.get_id = lambda x:[self.events2labels[i] for i in x]
        self.dataset.reset_index(inplace=True,drop=True)
        
        self.dataset = pd.read_csv('usound_mels/overall.csv')
        self.train = pd.read_csv('usound_mels/train.csv')
        self.test = pd.read_csv('usound_mels/val.csv')
    def _augment(self):
        pass
    
    def _settle_length(self, audio):
        
        if audio.shape[1]>self.chunk_length:
            audio = torch.split(audio,
                                split_size_or_sections=self.chunk_length,
                                dim=1)
            remainder_length = audio[-1].shape[-1]
            if remainder_length>self.chunk_thresh:
                audio[-1] = torch.nn.functional.pad(input = audio[-1],
                                  pad = (0,abs(self.chunk_length-remainder_length)),
                                  mode='constant',
                                  value=0)
                audio = torch.stack(audio,dim=0)
            else:
                audio = torch.cat(audio[0:-1],dim=0)
        
        elif audio.shape[1]<self.chunk_length:
            remainder_length = self.chunk_length-audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0,remainder_length))
                
            
        return audio
            
                            
    def _get_melspec(self,audio,sr,n_fft=1024,n_mels=128,
                     hop_length=400,f_min =15,f_max = 14000,power=1.5):
        transform = transforms.MelSpectrogram(sample_rate=sr,n_fft=n_fft,n_mels=n_mels,f_min =f_min,f_max = f_max,
                                              hop_length=hop_length,power=power).to(self.device)
        mel_specgram = transform(audio)
#         print(mel_specgram)
        return mel_specgram
    
    def _preprocess(self,fpath):
        audio,sr = torchaudio.load(os.path.join(self.data_path,fpath))
        audio = audio.to(self.device)
        
        if audio.shape[0]>1:
            audio = torch.unsqueeze(torch.mean(audio, dim=0),dim=0)
            
        if sr!=self.target_sr:
            resample = torchaudio.transforms.Resample(orig_freq= sr, new_freq = self.target_sr).to(self.device)           
            audio = resample(audio)
        
        audio[0,:] = audio[0,:]/abs(audio[0,:]).max()
        audio = self._settle_length(audio)
        return audio
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.train)
        elif self.mode == 'val':
            return len(self.test)
        else:
            return len(self.dataset)
    
    def __getitem__(self,idx):
        
        if self.mode == 'train':
            spec = torch.from_numpy(np.load(self.train.iloc[idx,1])).to(self.device)
            events = [torch.Tensor([self.train.iloc[idx,2]]).to(self.device),torch.Tensor([self.train.iloc[idx,3]]).to(self.device)]
        elif self.mode == 'val':
            spec = torch.from_numpy(np.load(self.test.iloc[idx,1])).to(self.device)
            events = [torch.Tensor([self.test.iloc[idx,2]]).to(self.device),torch.Tensor([self.test.iloc[idx,3]]).to(self.device)]
        else:
            spec = torch.from_numpy(np.load(self.dataset.iloc[idx,1])).to(self.device)
            events = [torch.Tensor([self.dataset.iloc[idx,2]]).to(self.device),torch.Tensor([self.dataset.iloc[idx,3]]).to(self.device)]
        return spec,events