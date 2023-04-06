import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split

import math
import numpy as np

import pandas as pd
import os
import csv

from tqdm import tqdm

from model01 import Lmodel

def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds

def select_feat(train_data,valid_data,test_data,select_all=True):
    y_train,y_valid=train_data[:,-1],valid_data[:,-1]
    raw_x_train,raw_x_valid,raw_x_test=train_data[:,:-1],valid_data[:,:-1],test_data
    if select_all:
        feat_idx=list(range(raw_x_test.shape[1]))
    else:
        feat_idx=[0,1,2,3,4]
    return raw_x_train[:,feat_idx],raw_x_valid[:,feat_idx],raw_x_test[:,feat_idx],y_train,y_valid


class nDataset(Dataset):
    def __init__(self,x,y=None):
        if y is None:
            self.y=y
        else:
            self.y=torch.FloatTensor(y)
        self.x=torch.FloatTensor(x)
    def __getitem__(self, ind):
        if self.y is None:
            return self.x[ind]
        else:
            return self.x[ind],self.y[ind]
    def __len__(self):
        return len(self.x)


def trainer(train_loader,valid_loader,model,config,device):
    cri=nn.MSELoss(reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(),lr=config['learning_rate'],momentum=0.9)

    if not os.path.isdir('./models'):
        os.makedirs('./models')

    best_loss,step,early_stop=math.inf,0,0
    for epoch in range(config['n_epochs']):
        model.train()
        loss_record=[]
        tqdm_train=tqdm(train_loader,position=0,leave=True)
        for x,y in train_loader:
            for x,y in tqdm_train:
                optimizer.zero_grad()
                x,y=x.to(device),y.to(device)
                pred=model(x)
                loss=cri(pred,y)
                loss.backward()
                step+=1
                loss_record.append(loss.detach().item())
                optimizer.step()

                tqdm_train.set_description(f"Epoch {epoch+1}/{config['n_epochs']}")
                tqdm_train.set_postfix({"loss":loss.detach().item()})        
        mean_train_loss=sum(loss_record)/len(loss_record)

        model.eval()
        loss_record=[]
        for x,y in valid_loader:
            x,y=x.to(device),y.to(device)
            with torch.no_grad():
                pred=model(x)
                loss=cri(pred,y)
            loss_record.append(loss.detach().item())
        
        mean_valid_loss=sum(loss_record)/len(loss_record)
        print(f"Epoch {epoch+1}/{config['n_epochs']}:mean_train_loss={mean_train_loss:.4f},mean_valid_loss={mean_valid_loss:.4f}")
        if mean_valid_loss < best_loss:
            best_loss=mean_valid_loss
            torch.save(model.state_dict(),config['save_path'])
            print(f"\n save best with loss {best_loss:.4f}")
            early_stop=0
        else:
            early_stop+=1
        
        if early_stop>config['early_stop']:
            print("\n not improve")
            return
        






device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,      # Your seed number, you can pick your lucky number. :)
    'select_all': True,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 30,     # Number of epochs.       used to be 3000     
    'batch_size': 256, 
    'learning_rate': 1e-5,              
    'early_stop': 400,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}

# Set seed for reproducibility
same_seed(config['seed'])


# train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days) 
# test_data size: 1078 x 117 (without last day's positive rate)
train_data, test_data = pd.read_csv('E:\\program\\vscode\\code\\homework001\\data\\covid.train.csv').values, pd.read_csv('E:\\program\\vscode\\code\\homework001\\data\\covid.test.csv').values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

# Print out the data size.
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# Select features
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

# Print out the number of features.
print(f'number of features: {x_train.shape[1]}')

train_dataset, valid_dataset, test_dataset = nDataset(x_train, y_train), \
                                            nDataset(x_valid, y_valid), \
                                            nDataset(x_test)

# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

model = Lmodel(input_dims=x_train.shape[1]).to(device)# put your model and data on the same computation device.
trainer(train_loader, valid_loader, model, config, device)