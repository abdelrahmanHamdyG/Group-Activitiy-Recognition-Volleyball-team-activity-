from torchvision import transforms as T
from dataloading.dataloader import get_data
import torch
from utils.logger import Logger
from modeling.player_level_classification.player_level_classification_model import PlayersModel,load_players_model
from collections import Counter
import torch.optim as optim 

import torch.nn as nn
from torch.utils.data import DataLoader
from modeling.b2.b2_model import load_model
from modeling.b2.b2_model import B2_Dataset
from dataloading.dataloader import save_checkpoint



logger=Logger("b2_model","modeling/b2/b2.log")

train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomApply([
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.5, hue=0.2)
    ,
    ], p=0.1),
    
    T.RandomGrayscale(p=0.2),
    T.RandomApply([
    T.RandomChoice([

    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    T.RandomRotation(degrees=8),
    T.RandomAffine(degrees=5, translate=(0.1, 0.1)),
    T.RandomPerspective(distortion_scale=0.2, p=0.5)
    ])
    ], p=0.1),

    

    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


test_transform = T.transforms.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((224, 224)),

    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def train():

    
    videos_train,annot_train, videos_val,annot_val, videos_test,annot_test=get_data(all_frames=True)

    train_labels = [ann.action for ann in annot_train]
    val_labels   = [ann.action for ann in annot_val]

    train_class_counts = Counter(train_labels)
    val_class_counts = Counter(val_labels)

    logger.info(f"Training class distribution: {train_class_counts}")
    logger.info(f"Validation class distribution: {val_class_counts}")

    players_model,_,_=load_players_model(classifier=False)
    for param in players_model.parameters():
        param.requires_grad = False
    

    
    train_dataset = B2_Dataset(videos_train, annot_train,train_transform,players_model)
    eval_dataset=B2_Dataset(videos_val,annot_val,test_transform,players_model)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=64,shuffle=False)

    
    model, optimizer, start_epoch = load_model(best_model=True,new_lr=0.0001,weight_decay=0.00006)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    
    
    
    
    


    criterion = nn.CrossEntropyLoss()

    
    scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.5)
    
    

    
    epochs=90
    
    for epoch in range(start_epoch+1,epochs,1):
        model.train()

        train_loss=0
        train_correct=0
        total=0
        logger.info (f"Epoch {epoch}, Current LR: {optimizer.param_groups[0]['lr']}")

        for idx,(inputs,output,_) in enumerate(train_loader):
            if idx%20==0 and idx:
                logger.info(f"Training: Epoch {epoch}, Batch {idx-1}/{len(train_loader)} loss {train_loss/(idx-1)} accuracy {train_correct/total}")
            
            inputs=inputs.to(device)
            output=output.to(device)
            predicted=model(inputs)
            loss=criterion(predicted,output)
            train_correct+=(torch.max(predicted,1)[1]==output).sum().item()
            total+=output.size(0)
            optimizer.zero_grad()
            loss.backward()
            train_loss+=loss.item()
            optimizer.step()
        
        logger.info(f"Epoch {epoch}, Loss: {train_loss / len(train_loader)} Accuracy: {train_correct / len(train_dataset)}")
        
        scheduler.step() 
       
        model.eval()
        eval_correct=0
        eval_loss=0
        total=0
        with torch.no_grad():
            for idx,(inputs,output,_) in enumerate(eval_loader):
                if idx%15==0 and idx:
                    logger.info(f"Evaluating : Epoch {epoch}, Batch {idx-1}/{len(eval_loader)} loss {eval_loss/(idx-1)} accuracy {eval_correct/total}")
                inputs=inputs.to(device)
                output=output.to(device)
                predicted=model(inputs)
                loss=criterion(predicted,output)
                eval_loss+=loss.item()
                total+=output.size(0)
                predicted_class=torch.max(predicted,1)[1]
                eval_correct+=(predicted_class==output).sum().item()
                

            
            logger.info(f"Eval Epoch {epoch}: Loss {eval_loss / len(eval_loader)}, Accuracy: {eval_correct / len(eval_dataset)}")
        save_checkpoint(epoch,model.state_dict(),optimizer.state_dict(),train_loss / len(train_loader),eval_loss / len(eval_loader),eval_correct / len(eval_dataset),"models_trained/b2/checkpoints","models_trained/b2/best_model.pth")
        
        
        
        
       

        







if __name__=="__main__":
    
    train()
