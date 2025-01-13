import torch.nn as nn 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import torchvision.models as models
from torchinfo import summary
from utils.logger import Logger
from torch.utils.data import Dataset


logger=Logger("b1_model","modeling/b1/b1.log")






class b1_net(nn.Module):
    def __init__(self):
        super(b1_net, self).__init__()

        self.resnet=models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
       
        self.resnet.fc=nn.Linear(self.resnet.fc.in_features,8)

    
    def forward(self,x):
        return self.resnet(x)



class B1_Dataset(Dataset):
    def __init__(self, image_pathes, annotations, transform):
        self.image_pathes = image_pathes
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, index):
        
        image = Image.open(self.image_pathes[index]).convert("RGB")
        
        
        
        if self.transform:
            image=self.transform(image)

        # Map action to index
        action_to_idx = {
            "r-set": 0, "l-set": 1, "r-winpoint": 2, "l-winpoint": 3,
            "l-pass": 4, "r-pass": 5, "l-spike": 6, "r-spike": 7
        }
        
        
        label = action_to_idx.get(self.annotations[index].action, 0)  # Default to 0 if action not found

        return image, label



def load_model(best_model=True,new_lr=None):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model=b1_net()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.00001,weight_decay=0.001)
    last_epoch=0
    current_path = os.getcwd()
    logger.info(f"Current path: {current_path}")
    files = os.listdir(current_path)
    logger.info(f"Files in current path: {files}")  
    if best_model:
        if os.path.exists("models_trained/b1/best_model.pth"):
            checkpoint_saved=torch.load("models_trained/b1/best_model.pth")
            logger.info(f"model loaded ")

            model.load_state_dict(checkpoint_saved["model_state_dict"],strict=False)
            optimizer.load_state_dict(checkpoint_saved["optimizer_state_dict"])
            last_epoch=checkpoint_saved["epoch"]
            print(last_epoch)
        else:
            logger.info("No model found starting from scratch")
    else:
        if os.path.exists("models_trained/b1/checkpoints"):
            checkpoints = [f for f in os.listdir("models_trained/b1/checkpoints") if f.endswith('.pth')]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('.')[0]))
                checkpoint_saved = torch.load(f"models_trained/b1/checkpoints/{latest_checkpoint}")
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
    return model,optimizer,last_epoch



def check_best_model_accuracy(checkpoint_path):
    if os.path.exists(checkpoint_path):    
        model_loaded = torch.load(checkpoint_path)
        return model_loaded["accuracy"]
    return 0


def predict_and_visualze(model,image_path, true_value,transform):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    predicted = model(image)
    action_to_idx = {
        0: "r-set", 1: "l-set", 2: "r-winpoint", 3: "l-winpoint",
        4: "l-pass", 5: "r-pass", 6: "l-spike", 7: "r-spike"
    }
    chosen_action = torch.argmax(predicted, dim=1).item()
    cv2.imshow(f"predicted {action_to_idx[chosen_action]} true_value {true_value}", cv2.imread(image_path))
    cv2.waitKey(0)
    logger.info(f"Predicted Action: {predicted} True Action: {true_value}")
