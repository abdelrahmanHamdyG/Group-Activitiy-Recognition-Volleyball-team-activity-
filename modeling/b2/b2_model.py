import dataloading.dataloader as dataloader
import numpy as np
import cv2
import os
from collections import Counter,defaultdict
import matplotlib.pyplot as plt 
from modeling.b1.b1_trainer import save_checkpoint
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from utils.player_level_classification import PlayersModel
from utils.logger import Logger
from torch.utils.data import Dataset,DataLoader,random_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


logger=Logger("b2_model")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class b2_dataset(Dataset):
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
        self.drop1 = nn.Dropout(0.5)
        
        # Second Fully Connected Layer
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.55)
        
        # Third Fully Connected Layer
        self.fc3 = nn.Linear(512, 8)
        

    def forward(self, x):
        # Aggregate player features using max pooling
        x = torch.mean(x, 1) # Shape: (batch_size, 2048)
        
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

def load_latest_checkpoint(model, optimizer, checkpoint_path="b2_checkpoints8", new_lr=None):
    if not os.path.exists(checkpoint_path) or len(os.listdir(checkpoint_path)) == 0:
        print("No checkpoints found.")
        return model, optimizer, 0
    
    latest_checkpoint = max(os.listdir(checkpoint_path), key=lambda x: int(x.split('.')[0]))
    checkpoint_file = os.path.join(checkpoint_path, latest_checkpoint)
    print(f"Loading checkpoint from {checkpoint_file}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    # Update learning rate if specified
    if new_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
            
        print(f"Learning rate updated to {new_lr}")
    
    return model, optimizer, epoch




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



        
def main():

    
    images,annot=dataloader.get_data()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet=PlayersModel(False)
    
    
    optimizer=optim.Adam(resnet.parameters(),lr=0.001)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    resnet,optimizer,start_epoch=load_latest_checkpoint(resnet,optimizer,checkpoint_path="player_level_classification_checkpoints3")
    resnet=resnet.to(device)
    



    dataset = b2_dataset(images, annot, transform, resnet)
    

    

    
    
    
    
    

    train_size = int(len(images) * 0.8)
    eval_size = int(len(images) * 0.1)
    test_size = len(images) - train_size - eval_size
    generator = torch.Generator().manual_seed(42)  
    train_data, eval_data, test_data = random_split(dataset, [train_size, eval_size, test_size],generator=generator)
    
    # log_class_counts(train_data, "Train")
    # log_class_counts(eval_data, "Eval")

    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=32)
    test_loader= DataLoader(test_data, batch_size=32)
    criterion = nn.CrossEntropyLoss()

    model = b2_net()
    model=model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002,weight_decay=0.01)
    scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.77)
    
    start_epoch = -1
    model, optimizer, start_epoch = load_latest_checkpoint(model, optimizer, checkpoint_path="b2_checkpoints8",new_lr=4.8145569800559e-05)
    desired_weight_decay = 0.001    

    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = desired_weight_decay


    model.train()
    epochs=150
    
    for epoch in range(start_epoch+1,epochs,1):
        epoch_loss=0
        correct=0
        total=0
        logger.info (f"Epoch {epoch}, Current LR: {optimizer.param_groups[0]['lr']}")

        for idx,(inputs,output,_) in enumerate(train_loader):
            if idx%40==1:
                logger.info(f"Epoch {epoch}, Batch {idx-1} accuracy {correct/total}")
            
            inputs=inputs.to(device)
            output=output.to(device)
            predicted=model(inputs)
            loss=criterion(predicted,output)
            correct+=(torch.max(predicted,1)[1]==output).sum().item()
            total+=output.size(0)
            optimizer.zero_grad()
            loss.backward()
            epoch_loss+=loss.item()
            optimizer.step()
        
        logger.info(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader)} Accuracy: {correct / total}")
        
        scheduler.step() 
       
        model.eval()
        correct=0
        eval_loss=0
        total=0
        with torch.no_grad():
            for idx,(inputs,output,_) in enumerate(eval_loader):
                if idx%40==1:
                    logger.info(f"Epoch {epoch}, Batch {idx-1} accuracy {correct/total}")
                inputs=inputs.to(device)
                output=output.to(device)
                predicted=model(inputs)
                loss=criterion(predicted,output)
                eval_loss+=loss.item()
                total+=output.size(0)
                predicted_class=torch.max(predicted,1)[1]
                correct+=(predicted_class==output).sum().item()
                

            accuracy = correct / len(eval_data)
            logger.info(f"Epoch {epoch} Eval Loss: {eval_loss / len(eval_loader)}, Accuracy: {accuracy:.4f}")
        model.train()
        save_checkpoint(epoch,model.state_dict(),optimizer.state_dict(),epoch_loss/len(train_loader),eval_loss/len(eval_loader),accuracy,"b2_checkpoints8")
        
       

        












if __name__ == "__main__":
    # main()

    counts_by_players = defaultdict(int)
    correct_by_players = defaultdict(int)

    
    images,annot=dataloader.get_data()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet=PlayersModel(False)
    
    
    optimizer=optim.Adam(resnet.parameters(),lr=0.001)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    resnet,optimizer,start_epoch=load_latest_checkpoint(resnet,optimizer,checkpoint_path="player_level_classification_checkpoints3")
    resnet=resnet.to(device)
    

    dataset = b2_dataset(images, annot, transform, resnet)
    
    
    model = b2_net()
    model=model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    

    model, optimizer = load_latest_checkpoint(model, optimizer, checkpoint_path="b2_checkpoints8")[:2]

    dataloader_instance=DataLoader(dataset,32,shuffle=False)
    
    train_size = int(len(images) * 0.8)
    eval_size = int(len(images) * 0.1)
    test_size = len(images) - train_size - eval_size
    generator = torch.Generator().manual_seed(42)  
    train_data, eval_data, test_data = random_split(dataset, [train_size, eval_size, test_size],generator=generator)
    
    # log_class_counts(train_data, "Train")
    # log_class_counts(eval_data, "Eval")

    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=32)
    test_loader= DataLoader(test_data, batch_size=32)




    model.eval()
    correct=0
    eval_loss=0
    criterion = nn.CrossEntropyLoss()
    all_labels=[]
    all_preds=[]

    total=0
    with torch.no_grad():
        for idx,(inputs,output,no_of_players) in enumerate(eval_loader):
            if idx%40==1:
                logger.info(f" Batch {idx-1} accuracy {correct/total}")
            inputs=inputs.to(device)
            
            output=output.to(device)
            predicted=model(inputs)
            loss=criterion(predicted,output)
            eval_loss+=loss.item()
            total+=output.size(0)
            predicted_class=torch.max(predicted,1)[1]
            correct+=(predicted_class==output).sum().item()
            for i in range(inputs.size(0)):
                players_count=no_of_players[i].item()
                counts_by_players[players_count]+=1
                if predicted_class[i].item()==output[i].item():
                    correct_by_players[players_count] += 1


            all_labels.extend(output.cpu().numpy())
            all_preds.extend(predicted_class.cpu().numpy())

        accuracy = correct /total
        logger.info(f"  Loss: {eval_loss / len(eval_loader)}, Accuracy: {accuracy:.4f}")
    
    logger.info("Accuracy by number of players:")
    for n_players in sorted(counts_by_players.keys()):
        total_n = counts_by_players[n_players]
        correct_n = correct_by_players[n_players]
        logger.info(f"{n_players} players: {correct_n}/{total_n} = {correct_n/total_n:.4f}")


    cm=confusion_matrix(all_labels,all_preds)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[ "r-set", "l-set", "r-winpoint", "l-winpoint",
            "l-pass", "r-pass", "l-spike", "r-spike"])
    disp.plot(cmap="Blues")
    plt.show()

    









