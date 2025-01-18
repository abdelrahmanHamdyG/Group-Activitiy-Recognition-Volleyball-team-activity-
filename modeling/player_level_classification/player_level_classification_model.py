from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn 
import torchvision.models as models
import torch
from utils.logger import Logger
import os 


logger=Logger("player_level_classification_model","modeling/player_level_classification/player_level_classification.log")

class PlayerDataset(Dataset):
    def __init__(self,players_images_paths,players_annotations,transform):
        self.transform=transform   
        self.players_images_paths=players_images_paths
        self.players_annotations=players_annotations
    
    def __len__(self):
        return len(self.players_images_paths)
    
    def __getitem__(self, index):
        image=Image.open(self.players_images_paths[index])
        image=self.transform(image)
        class_to_index = {
            "waiting": 0,
            "setting": 1,
            "digging": 2,
            "falling": 3,
            "spiking": 4,
            "blocking": 5,
            "jumping": 6,
            "moving": 7,
            "standing": 8
        }
        return image,class_to_index[self.players_annotations[index]]

class PlayersModel(nn.Module):
    def __init__(self, classifier=True):
        super(PlayersModel, self).__init__()
        self.classifier = classifier
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        self.resnet = nn.Sequential(*list(resnet50.children())[:-1])
        self.fc = nn.Linear(2048, 9)
        self.dropout = nn.Dropout(0.5)
        
        for param in resnet50.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = True
        for param in resnet50.layer4.parameters():
            param.requires_grad = True
        for param in resnet50.layer3.parameters():
            param.requires_grad = True
        
        
        

    def forward(self, x):
        x = self.resnet(x)
        if not self.classifier:
            
            return x
            
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def load_players_model(best_model=True,new_lr=None,weight_decay=None,classifier=False):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model=PlayersModel(classifier=classifier)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0001,weight_decay=0.005)
    last_epoch=0
    current_path = os.getcwd()
    logger.info(f"Current path: {current_path}")
    files = os.listdir(current_path)
    logger.info(f"Files in current path: {files}")  
    if best_model:
        if os.path.exists("models_trained/player_level_classification/best_model.pth"):
            checkpoint_saved=torch.load("models_trained/player_level_classification/best_model.pth")
            logger.info(f"model loaded ")

            model.load_state_dict(checkpoint_saved["model_state_dict"],strict=False)
            optimizer.load_state_dict(checkpoint_saved["optimizer_state_dict"])
            last_epoch=checkpoint_saved["epoch"]
            
        else:
            logger.info("No model found starting from scratch")
    else:
        if os.path.exists("models_trained/player_level_classification/checkpoints"):
            checkpoints = [f for f in os.listdir("models_trained/player_level_classification/checkpoints") if f.endswith('.pth')]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('.')[0]))
                checkpoint_saved = torch.load(f"models_trained/player_level_classification/checkpoints/{latest_checkpoint}")
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
    model.classifier=classifier
    return model,optimizer,last_epoch



