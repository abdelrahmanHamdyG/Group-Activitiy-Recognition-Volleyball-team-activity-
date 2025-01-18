
import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from torchvision import transforms as T
import torch
from utils.eval_metrics import display_confusion_matrix
import os

import torch.nn as nn
import torch.optim as optim
from utils.logger import Logger
import dataloading.dataloader as dataloader
from dataloading.dataloader import save_checkpoint
from torch.utils.data import  DataLoader
from modeling.b1.b1_model import B1_Dataset, load_model
from collections import Counter

logger = Logger("b1_trainer","modeling/b1/b1.log")



train_transform = T.transforms.Compose([
    T.transforms.Resize((256, 256)),
    T.transforms.CenterCrop((224, 224)),
    T.RandomApply([
        T.RandomChoice([
            T.GaussianBlur(kernel_size=3),
            T.ColorJitter(brightness=0.2),
        ])
    ], p=0.85),
    T.RandomApply([
        T.ColorJitter(
            brightness=0.2,     
            contrast=0.0,       
            saturation=0.3,     
            hue=0.06            
        )
    ], p=0.15),
        


    T.transforms.ToTensor(),
    T.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = T.transforms.Compose([
    T.transforms.Resize((256, 256)),
    T.transforms.CenterCrop((224, 224)),
    T.transforms.ToTensor(),
    T.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])









# Main Training Loop
def train():
    
    videos_train,annot_train, videos_val,annot_val, videos_test,annot_test=dataloader.get_data(all_frames=True)
    train_labels = [ann.action for ann in annot_train]
    val_labels   = [ann.action for ann in annot_val]

    train_class_counts = Counter(train_labels)
    val_class_counts = Counter(val_labels)

    logger.info(f"Training class distribution: {train_class_counts}")
    logger.info(f"Validation class distribution: {val_class_counts}")
    train_size=len(videos_train)
    eval_size=len(videos_val)
    train_dataset = B1_Dataset(videos_train, annot_train,train_transform)
    print(train_size)
    print(eval_size)
    eval_dataset=B1_Dataset(videos_val,annot_val,eval_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=3)
    eval_loader = DataLoader(eval_dataset, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, optimizer, start_epoch = load_model(True,0.00001)

    
    criterion = nn.CrossEntropyLoss()

    
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode="min",factor=0.1,patience=3,verbose=True)

    
    epochs = 15
    for epoch in range(start_epoch+1, epochs):
        train_loss = 0
        model.train()
        correct = 0
        logger.info(f"Training: Epoch {epoch}, Current LR: {optimizer.param_groups[0]['lr']}")

        for idx, (inputs, output) in enumerate(train_loader):
            
            
            

            if idx and idx % 20 == 0:
                logger.info(f"Epoch {epoch}, Batch {idx}/{len(train_loader)} loss: {train_loss / idx} accuracy {correct / (idx * 64)}")
            
            inputs, output = inputs.to(device), output.to(device)
            optimizer.zero_grad()
            predicted = model(inputs)
            predicted_class=torch.argmax(predicted,dim=1)
            correct += (predicted_class == output).sum().item()
            loss = criterion(predicted, output)            
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
        
        logger.info(f"Train: Epoch {epoch} loss: {train_loss / len(train_loader)} Accuracy {correct / train_size}\n-------------------------")

        # Evaluation
        model.eval()
        eval_loss = 0
        correct = 0
        all_predicted=[]
        all_labels=[]
        logger.info(f"Evaluating: Epoch {epoch}")
        with torch.no_grad():
            for idx, (inputs, output) in enumerate(eval_loader):
                if idx and idx % 10 == 0:
                    logger.info(f"Eval Batch {idx}/{len(eval_loader)} loss: {eval_loss / idx} accuracy {correct / (idx * 64)}")
                    
                
                inputs, output = inputs.to(device), output.to(device)
                predicted = model(inputs)
                chosen_action = torch.argmax(predicted, dim=1)
                all_predicted.extend(chosen_action.cpu().numpy())
                all_labels.extend(output.cpu().numpy())
                correct += (chosen_action == output).sum().item()
                loss = criterion(predicted, output)
                eval_loss += loss.item()

        accuracy = correct / eval_size
        logger.info(f"Eval: Epoch {epoch}  Loss: {eval_loss /eval_size}, Accuracy: {accuracy:.4f}\n---------------------------------------\n\n")
        scheduler.step(eval_loss / len(eval_loader))
        
        save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(),
                        train_loss / len(train_loader), eval_loss / len(eval_loader), accuracy, checkpoint_path="models_trained/b1/checkpoints")
        
        # display_confusion_matrix(all_predicted,all_labels,["r-set", "l-set", "r-winpoint", "l-winpoint", "l-pass", "r-pass", "l-spike", "r-spike"],logger)
   
        

def test_and_eval():
    model, _, _ = load_model(True)

    videos_train,annot_train, videos_val,annot_val, videos_test,annot_test=dataloader.get_data(all_frames=False)

    eval_dataset = B1_Dataset(videos_val, annot_val, eval_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=64)
    test_dataset = B1_Dataset(videos_test, annot_test, eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=64)

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():

        test_correct = 0
        all_predicted=[]
        all_labels=[]
        test_loss=0
        model.eval()
        for idx, (inputs, output) in enumerate(test_loader):
            inputs, output = inputs.to(device), output.to(device)
            predicted = model(inputs)
            chosen_action = torch.argmax(predicted, dim=1)
            all_predicted.extend(chosen_action.cpu().numpy())
            all_labels.extend(output.cpu().numpy())
            test_correct += (chosen_action == output).sum().item()
            loss = criterion(predicted, output)
            test_loss += loss.item()
        
        eval_loss=0
        eval_correct=0   
        for idx, (inputs, output) in enumerate(eval_loader):
            inputs, output = inputs.to(device), output.to(device)
            predicted = model(inputs)
            chosen_action = torch.argmax(predicted, dim=1)
            all_predicted.extend(chosen_action.cpu().numpy())
            all_labels.extend(output.cpu().numpy())
            eval_correct += (chosen_action == output).sum().item()
            loss = criterion(predicted, output)
            eval_loss += loss.item()
        


    logger.info(f"Test: Loss: {test_loss / len(test_loader)}, Accuracy: {test_correct / len(videos_test)}")
    logger.info(f"Eval: Loss: {eval_loss / len(eval_loader)}, Accuracy: {eval_correct / len(videos_val)}")
    

    display_confusion_matrix(all_predicted,all_labels,["r-set", "l-set", "r-winpoint", "l-winpoint", "l-pass", "r-pass", "l-spike", "r-spike"],logger)

    


if __name__ == "__main__":
    # train()
    test_and_eval()
