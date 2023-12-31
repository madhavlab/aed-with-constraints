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


from constraints_code import Constraints
from model import SED
from get_data import getdata
from utils import make_one_hot, step, evaluate



os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda')

num_epochs = 130
batch_size = 80
epsilon = 0.1
save_path = 'inference_results_settings_final12'
train = getdata(mode = 'train', device = device,
                load_existing = False,save_path = save_path)
val = getdata(mode = 'val', device = device,
                load_existing = False,save_path = save_path)
sampler = torch.utils.data.RandomSampler(data_source=train,
                                        replacement=False,
                                        num_samples=None, 
                                        generator=None)
train_dl = DataLoader(train,batch_size=batch_size, drop_last=True,sampler=sampler)
dd_train_dl = DataLoader(train,batch_size=batch_size, shuffle=False, drop_last=True,sampler=sampler)
val_dl = DataLoader(val,batch_size=batch_size, shuffle=False, drop_last=True)
num_batches = [len(train_dl),len(dd_train_dl),len(val_dl)]

warmup_iters = int(10*num_batches[0])#8epochs

initial_lr_w = 0.003
max_lr_w = 2.5*initial_lr_w
min_lr_w = 1e-2*initial_lr_w

initial_lr_dd = 0.0035
beta = 0.01

initial_loss,current_loss = 0.0,0.0
t = 1
t1 = 1
num_iter = 0
lambda_iters = 0
total_iters = int(num_batches[0]*num_epochs)#120 epochs
evaluate_after = int(num_batches[0]*3) #6epochs
l = 2 # the number of iterations (l) to wait before dd are updated
d = 50 # the increment in number of iterations (l) to wait before dd are updated
last_lambda_iter = 0
lambda_batch_idx = 0
lambda_iter = 0
last_logged_eval = 0
last_lambda_update = 0
w_weight_bce,w_weight_ce,dd_constraint_wt = 1.1,1.6,0.85

model = SED().to(device)
constraints = Constraints(ontology = train.ontology).to(device)
criterion_bce = nn.BCEWithLogitsLoss(weight=None,
                                    reduction='mean',
                                    pos_weight=torch.Tensor([1.5,2.1,1.8,1.5,1.5,1.5,1.5,
                                                            1.5,1.5,2.1,1.1,1.1,1.1,1.1])).to(device)
criterion_ce = nn.CrossEntropyLoss(reduction='mean',
                                label_smoothing=0.0)                                          
optimizer_w = torch.optim.Adam(model.parameters(),
                            lr=initial_lr_w)
optimizer_dd = torch.optim.Adam(constraints.parameters(),
                                lr=initial_lr_dd)
scheduler_w = torch.optim.lr_scheduler.OneCycleLR(optimizer = optimizer_w,
                                                max_lr = max_lr_w,
                                                epochs = num_epochs,
                                                steps_per_epoch = num_batches[0],
                                                anneal_strategy='cos',
                                                pct_start=0.3,
                                                cycle_momentum=True,
                                                base_momentum=0.85,
                                                max_momentum=0.95,
                                                div_factor = max_lr_w/initial_lr_w,
                                                final_div_factor = initial_lr_w/min_lr_w,
                                                three_phase=False,
                                                last_epoch=-1,
                                                verbose=False)

scheduler_dd = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer_dd,
                                                lr_lambda = lambda t: 1/(1+(beta*t)),
                                                last_epoch=-1,
                                                verbose=False)
    

h_k = pd.DataFrame(np.zeros((evaluate_after,len(train.ontology.keys()))),columns=[f'constraint{k}' for k in range(len(train.ontology.keys()))])
training_loss = pd.DataFrame(np.zeros((num_epochs,4)),columns=['bce','ce','cons','total'])

torch.cuda.empty_cache()
epoch = 0
while epsilon*initial_loss<=current_loss and num_iter<total_iters:
    w_loss_bce = 0.0
    w_loss_ce = 0.0
    c_loss = 0.0
    tot_loss = 0.0
    for mel_spec, target_c in tqdm(train_dl):
        model.train()
        constraints.train()
        mel_spec,target_c,target_p = mel_spec.to(device),target_c[0].to(device),target_c[1].to(device)
        logits = model(mel_spec)
#         logits_children,logits_parents
        loss_w_bce = criterion_bce(logits,make_one_hot(target_c,target_p))
        loss_w_ce = criterion_ce(logits[:,:10],target_c.squeeze(1).long())
        for lambda_param in constraints.parameters():
            lambda_param.requires_grad = False
        closs,hk = constraints(torch.sigmoid(logits))
        
        train_loss = w_weight_bce*loss_w_bce + w_weight_ce*loss_w_ce + dd_constraint_wt*closs 
        
        w_loss_bce += loss_w_bce.item()
        w_loss_ce += loss_w_ce.item()
        c_loss += closs.item()
        tot_loss += train_loss.item()
        h_k.iloc[num_iter-int(last_logged_eval*evaluate_after),:] = hk.tolist()
        
        step(model = model,
            loss = train_loss,
            optimizer = optimizer_w,
            scheduler = scheduler_w)
        
        num_iter += 1

        if ((num_iter+1) % evaluate_after == 0):
            last_logged_eval += 1
#             print(f'Evaluating-{last_logged_eval}th time. Total iters so far-{num_iter+1}')
            evaluate(model = model,
                    constraints = constraints,
                    dataset = val,
                    label2id = val.unique_labels,
                    thresh=0.5,
                    save_path = save_path,
                    iteration_name = f'eval-{last_logged_eval}-totaliters-{num_iter+1}')
            training_loss.to_csv(os.path.join(save_path,'train_loss.csv'))
            h_k.to_csv(os.path.join(save_path,f'h_k_after_{num_iter+1}iters.csv'))
#             print('Evaluation done.')
        if (num_iter >= warmup_iters) and (num_iter-last_lambda_iter >= l):
#             print('updating lambda')
            model.eval()
            constraints.train()
            for lambda_param in constraints.parameters():
                lambda_param.requires_grad = True
            for idx,minibatch in enumerate(dd_train_dl):
                with torch.no_grad():
                    logits= model(minibatch[0])
                closs,_ = constraints(torch.sigmoid(logits).detach())
                step(model = constraints,
                    loss = -1.0*dd_constraint_wt*closs,
                    optimizer = optimizer_dd,
                    scheduler = scheduler_dd)

                t+=1
                last_lambda_iter = num_iter
                l += d
                break
            model.train()
    training_loss.iloc[epoch] = [w_loss_bce/num_batches[0],w_loss_ce/num_batches[0],c_loss/num_batches[0],tot_loss/num_batches[0]]
    epoch += 1    