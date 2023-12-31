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



def make_one_hot(A, B, device=torch.device('cuda')):
    identity_A = torch.eye(10).to(device)
    identity_B = torch.eye(4).to(device)
    
    multihot_encoded_A = identity_A[A.flatten().long()]
    multihot_encoded_B = identity_B[B.flatten().long() - 10]
    
    C = torch.cat((multihot_encoded_A, multihot_encoded_B), dim=1)
    return C

def step(model, loss, optimizer, scheduler = False, clip_val = None,grad_norm = None):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()

def evaluate(model,constraints,dataset,label2id,thresh=0.5,
             device=torch.device('cuda'),save_path = 'inference_results',
             iteration_name = 'iter'):
    y_true_l1 = torch.zeros(len(dataset),)
    y_pred_l1 = torch.zeros(len(dataset),)
    y_true_l2 = torch.zeros(len(dataset),)
    y_pred_l2 = torch.zeros(len(dataset),)                    
    
    model.eval()
    constraints.eval()
    for lambda_param in constraints.parameters():
        lambda_param.requires_grad = False
    estimated_tag_list = dcase_util.containers.MetaDataContainer()
    reference_tag_list = dcase_util.containers.MetaDataContainer()
    with torch.no_grad():
        ended_at = 0
        for i in tqdm(range(len(dataset))):
            mel_spec, target = dataset[i]
            mel_spec = torch.unsqueeze(mel_spec.to(device),dim=0)
            logits = model(mel_spec)
            pred,targets = torch.sigmoid(logits)[0,:], make_one_hot(target[0],target[1]).to(device)
            y_pred_l1[i] = torch.argmax(pred[:10], dim=0).item()
            y_pred_l2[i] = torch.argmax(pred[10:], dim=0).item()
            y_true_l2[i] = torch.where(targets[0,10:]==1)[0].item()
            y_true_l1[i] = torch.where(targets[0,:10]==1)[0].item()
            
            _,h_k = constraints(torch.unsqueeze(pred,dim=0))
            fname = f'test{i+np.random.randint(1000)}.wav'
            pred = list(label2id.iloc[torch.where(pred>thresh)[0].cpu().tolist(),0])
            targets = list(label2id.iloc[torch.where(targets[0,:]==1)[0].cpu().tolist(),0])
            estimated_tag_list.append({
                'filename':fname,
                'tags':pred
            })
            reference_tag_list.append({
                'filename':fname,
                'tags':targets
            })         

      
    tag_evaluator = get_metrics_sedeval(tags = reference_tag_list.unique_tags)
    tag_evaluator.evaluate(reference_tag_list=reference_tag_list,
                           estimated_tag_list=estimated_tag_list)
    multilabel_metrics = tag_evaluator.results()
    save_path = os.path.join(save_path,iteration_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(multilabel_metrics,os.path.join(save_path,'multilabel_metrics.pth'))
    torch.save({'level1':{'true':y_true_l1,'pred':y_pred_l1},'level2':{'true':y_true_l2,'pred':y_pred_l2}},
               os.path.join(save_path,'predictions_gt.pth'))
    torch.save(model.state_dict(),os.path.join(save_path,'model.pth'))
    torch.save(constraints.state_dict(),os.path.join(save_path,'constraints.pth'))