from torch import nn,Tensor, stack, int64,float32, float64, argmax, save
import torch
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from utils import *
from Network import *
import os
from tqdm import tqdm
def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        pass

def train(net, train_loader, valid_loader, epoch, lossf, optimizer, DEVICE, save_path):
    history = {'loss': [], 'acc': [], 'precision': [], 'f1score': [], "recall": []}
    for e in range(epoch):
        net.train()
        for sample in tqdm(train_loader, desc="train: "):
            X =  torch.stack([s["x"] for s in sample], dim=0)
            Y = torch.stack([s["label"] for s in sample], dim=0)
            out = net(X.squeeze().type(float32).to(DEVICE))
            # print(out.size())
            loss = lossf(out.squeeze().type(float32).to(DEVICE), Y.squeeze().type(float32).to(DEVICE))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if valid_loader is not None:
            net.eval()
            print("valid start")
            with torch.no_grad():
                for key, value in valid(net, valid_loader, e, lossf, DEVICE).items():
                    history[key].append(value)
        if save_path is not None:            
            save(net.state_dict(), f"./Models/{save_path}/net.pt")
    if valid_loader is not None:                    
        return history
    else:
        return None
    
def valid(net, valid_loader, e, lossf, DEVICE):
    length = len(valid_loader)
    loss=0
    for sample in tqdm(valid_loader, desc="validation:"):
        X =  torch.stack([s["x"] for s in sample])
        Y = torch.stack([s["label"] for s in sample])
        out = net(X.type(float32).squeeze().to(DEVICE))
        loss += lossf(out.squeeze().type(float32).to(DEVICE), Y.squeeze().type(float32).to(DEVICE))
        
    return {'loss': loss.item()/length}