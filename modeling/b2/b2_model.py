import dataloading.dataloader as dataloader
import numpy as np
import cv2
import os
from collections import Counter,defaultdict
import matplotlib.pyplot as plt 
from modeling.b1.b1_trainer import save_checkpoint
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from modeling.player_level_classification.player_level_classification_model import PlayersModel
from utils.logger import Logger
from torch.utils.data import Dataset,DataLoader,random_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


logger=Logger("b2_model","modeling/b2/b2.log")



class B2_Dataset(Dataset):
    def __init__(self,images,annot,transform,model):
        self.images=images
        self.annot=annot
        self.transform=transform
        self.model=model
    
    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, index):
        
        players_positions=self.annot[index] .players
        original_image=cv2.imread(self.images[index])
        cropped_image_list=[]
        for player in players_positions:
            (x,y,width,height)=(player.x1,player.y1,player.w,player.h)
            cropped_image=original_image[y:y+height,x:x+width]
            cropped_image=Image.fromarray(cropped_image)
            cropped_image=self.transform(cropped_image)
            cropped_image_list.append(cropped_image)
        

        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            cropped_image_list=torch.stack(cropped_image_list).to(device)
            
            cropped_image_list=self.model(cropped_image_list)
            
            
            
        
        
        cropped_image_list=cropped_image_list.squeeze(2).squeeze(2)
        
        no_of_players=cropped_image_list.shape[0]
        if(no_of_players<12):
            cropped_image_list=torch.cat([cropped_image_list,torch.zeros(12-cropped_image_list.shape[0],2048).to(device)],0)
            

        action_to_idx = {
            "r-set": 0, "l-set": 1, "r-winpoint": 2, "l-winpoint": 3,
            "l-pass": 4, "r-pass": 5, "l-spike": 6, "r-spike": 7
        }
        return cropped_image_list, action_to_idx[self.annot[index].action],no_of_players
            




            

class b2_net(nn.Module):
    def __init__(self):
        super(b2_net, self).__init__()
        
        # First Fully Connected Layer
        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.45)
        
        # Second Fully Connected Layer
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()

        self.drop2 = nn.Dropout(0.5)
        
        # Third Fully Connected Layer
        self.fc3 = nn.Linear(512, 8)
        

    def forward(self, x):
        # Aggregate player features using max pooling
        x,_= torch.max(x, 1) # Shape: (batch_size, 2048)
        
        # First Layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        
        # Second Layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        
        return x



def load_model(best_model=True,new_lr=None,weight_decay=None):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model=b2_net()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0001,weight_decay=0.001)
    last_epoch=0
    current_path = os.getcwd()
    logger.info(f"Current path: {current_path}")
    files = os.listdir(current_path)
    logger.info(f"Files in current path: {files}")  
    if best_model :
        if os.path.exists("models_trained/b2/best_model.pth") :
            checkpoint_saved=torch.load("models_trained/b2/checkpoints/30.pth")
            logger.info(f"model loaded ")

            model.load_state_dict(checkpoint_saved["model_state_dict"],strict=False)
            optimizer.load_state_dict(checkpoint_saved["optimizer_state_dict"])
            last_epoch=checkpoint_saved["epoch"]
            
        else:
            logger.info("No model found starting from scratch")
    else:
        if os.path.exists("models_trained/b2/checkpoints"):
            checkpoints = [f for f in os.listdir("models_trained/b2/checkpoints") if f.endswith('.pth')]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('.')[0]))
                checkpoint_saved = torch.load(f"models_trained/b2/checkpoints/{latest_checkpoint}")
                logger.info(f"Loaded checkpoint {latest_checkpoint}")
                

                model.load_state_dict(checkpoint_saved["model_state_dict"])
                optimizer.load_state_dict(checkpoint_saved["optimizer_state_dict"])
                last_epoch = checkpoint_saved["epoch"]
            else:
                logger.info("No checkpoints found, starting from scratch")
        else:
            logger.info("No model found starting from scratch")
    
    model=model.to(device)
    for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
    if new_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    if weight_decay:
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = weight_decay
    return model,optimizer,last_epoch




def log_class_counts(dataset_split, split_name):
    # Extract labels from the dataset split
    labels = [dataset_split.dataset[i][1] for i in dataset_split.indices]
    
    # Count occurrences of each class
    class_counts = Counter(labels)
    
    # Log the class distribution
    logger.info(f"{split_name} Class Counts:")
    for cls in range(8):  # Assuming 8 classes indexed from 0 to 7
        count = class_counts.get(cls, 0)
        logger.info(f"  Class {cls}: {count}")



