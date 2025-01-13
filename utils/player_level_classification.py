import dataloading.dataloader as dataloader
import cv2
import torch
import pickle
from modeling.b1.b1_trainer import save_checkpoint
import os
import torch.nn as nn 
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from PIL import Image
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset,random_split,WeightedRandomSampler
from torchvision import transforms
from utils.logger import Logger

logger=Logger("player_level_classification")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_latest_checkpoint(model, optimizer, checkpoint_path, new_lr=None):
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
    for param in model.resnet[7].parameters():
        param.requires_grad = True






    return model, optimizer, epoch

class PlayerDataset(Dataset):
    def __init__(self,players_images_paths,players_annotations):
        self.players_images_paths=players_images_paths
        self.players_annotations=players_annotations
    
    def __len__(self):
        return len(self.players_images_paths)
    
    def __getitem__(self, index):
        image=Image.open(self.players_images_paths[index])
        image=transform(image)
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
    


import dataloading.dataloader as dataloader
import cv2
import torch
import pickle
from modeling.b1.b1_trainer import save_checkpoint
import os
import torch.nn as nn 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import transforms
from utils.logger import Logger

logger = Logger("player_level_classification")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PlayerDataset(Dataset):
    def __init__(self, players_images_paths, players_annotations):
        self.players_images_paths = players_images_paths
        self.players_annotations = players_annotations
    
    def __len__(self):
        return len(self.players_images_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.players_images_paths[index])
        image = transform(image)
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
        return image, class_to_index[self.players_annotations[index]]

class PlayersModel(nn.Module):
    def __init__(self, classifier=True):
        super(PlayersModel, self).__init__()
        self.classifier = classifier
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze all ResNet layers first
        for param in resnet50.parameters():
            param.requires_grad = False
        
        # Unfreeze layer4 (last ResNet block)
        for param in resnet50.layer4.parameters():
            param.requires_grad = True
        for param in resnet50.layer3.parameters():
            param.requires_grad = True
        
        
        self.resnet = nn.Sequential(*list(resnet50.children())[:-1])
        self.fc = nn.Linear(2048, 9)

        # Ensure fc is trainable
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.resnet(x)
        if not self.classifier:
            return x
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_players_data():
    logger.info("loading players data")
    with open("./players_dataset/actions.pkl", 'rb') as all_actions:
        actions=pickle.load(all_actions)
    logger.info("actions are loaded")
    with open("./players_dataset/players_images_paths.pkl", 'rb') as all_images:
        images=pickle.load(all_images)
    logger.info("images are loaded")
    
    return images,actions

def get_balanced_sampler( annotations):
    # Provided class occurrences
    class_counts = torch.bincount(torch.tensor(annotations))
    logger.info(f"class counts are {class_counts}")

   
    # Calculate class weights (inverse of frequency)
    class_weights = 1.0 / (class_counts**0.7)
    logger.info(f"class weights are {class_weights}")
    # Map annotations to class weights
    sample_weights = [class_weights[label] for label in annotations]
    
    # Create the sampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    return sampler


def diplay_confusion_matrix(model,images,annot):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    all_data_set=PlayerDataset(images,annot)
    all_data_loader=DataLoader(all_data_set,batch_size=16,shuffle=False)

    all_predicted=[]
    correct=0
    total=0
    all_labels=[]
    idx=0
    for inputs,outputs in all_data_loader:
        idx+=1
        inputs,outputs=inputs.to(device),outputs.to(device)
        predicted=model(inputs)
        
        chosen_action=torch.argmax(predicted,dim=1)
        number_of_correct=(chosen_action==outputs).sum().item()
        correct+=number_of_correct
        total+=outputs.size(0)
        incorrect_labels=outputs[chosen_action!=outputs]
        predicted_incorrect=chosen_action[chosen_action!=outputs]
        all_labels.extend(outputs.cpu().numpy())
        all_predicted.extend(chosen_action.cpu().numpy())
        logger.info(f"batch number {idx}  number of correct is {number_of_correct}/16  incorrect labels are {incorrect_labels} predicted incorrect are {predicted_incorrect}")

    logger.info(f"accuracy is {correct/total}")
    cm=confusion_matrix(all_labels,all_predicted)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["waiting","setting","digging","falling","spiking","blocking","jumping","moving","standing"])
    
    disp.plot(cmap="Blues")
    plt.show()



def main():
    dataloader.extract_players_images()
    print("images are extracted")
    images,actions=get_players_data()
    train_size=int(0.9*len(images))
    test_size=len(images)-train_size
    
    generator = torch.Generator().manual_seed(42)  

    data_set=PlayerDataset(images,actions)
    
    train_dataset, test_dataset = random_split(data_set, [train_size, test_size], generator=generator)
    
    
    logger.info("datasets are created")
    logger.info(f"train dataset size is {len(train_dataset)} and test dataset size is {len(test_dataset)}")
    
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    logger.info("train labels are created")
    # Create the sampler using the actual labels in train_dataset
    sampler = get_balanced_sampler(train_labels)
    logger.info("sampler is created")
    train_loader = DataLoader(train_dataset, batch_size=16,sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model=PlayersModel()
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_epoch=0
    model,optimizer,start_epoch=load_latest_checkpoint(model,optimizer,checkpoint_path="player_level_classification_checkpoints3",new_lr=0.00002)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


    model=model.to(device)
    
    scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=6,gamma=0.2)

    
    for epoch in range(start_epoch+1,55,1):
        model.train()
        epoch_loss=0
        total=1
        correct=1
        for idx,(images,labels) in enumerate(train_loader):
            if idx%20==0:
                logger.info(f"epoch {epoch} training for batch {idx} learning rate is {optimizer.param_groups[0]['lr']} accuracy till now = {correct/total}")
                logger.info(f" outputs is {labels}")
            
            images=images.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            predicted_class=torch.argmax(outputs,dim=1)
            correct+=(predicted_class==labels).sum().item()
            total+=labels.size(0)
            loss=criterion(outputs,labels)
            epoch_loss+=loss.item()
            loss.backward()
            optimizer.step()
        logger.info(f"loss of epoch {epoch} is {epoch_loss/len(train_loader)} accuracy is {correct/total}")
        scheduler.step()
        model.eval()    
        with torch.no_grad():
            correct=1
            total=1
            eval_loss=0
            for idx,(images,labels) in enumerate(test_loader):
                if idx%20==0:
                    logger.info(f"epoch {epoch} eval for batch {idx} learning rate is {optimizer.param_groups[0]['lr']} accuracy till now = {correct/total}")
                
                images=images.to(device)
                labels=labels.to(device)
                outputs=model(images)
                _,predicted=torch.max(outputs.data,1)
                loss=criterion(outputs,labels)
                eval_loss+=loss.item()
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()
            accuracy=correct/total
            logger.info(f"eval accuracy of epoch {epoch} is {accuracy} and loss is {eval_loss/len(test_loader)}")
        save_checkpoint(epoch,model.state_dict(),optimizer.state_dict(),epoch_loss/len(train_loader),eval_loss/len(test_loader),accuracy,"player_level_classification_checkpoints3")



        


# 
    


if __name__ == "__main__":
    main()
    
    

    # model=PlayersModel()
    
    
    # optimizer=optim.Adam(model.parameters(),lr=0.001)
    # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # start_epoch=0
    # model,optimizer,start_epoch=load_latest_checkpoint(model,optimizer,checkpoint_path="player_level_classification_checkpoints3")
    # model=model.to(device)

    # images,actions=get_players_data()
    # diplay_confusion_matrix(model,images,actions)

    # exit()
    # img1=cv2.imread(images[1623])
    # img2=cv2.imread(images[752])
    # img3=cv2.imread(images[3248])

    # img1_tr=transform(Image.open(images[1623]))
    # img2_tr=transform(Image.open(images[752]))
    # img3_tr=transform(Image.open(images[3248]))

    # img1_tr=img1_tr.unsqueeze(0).to(device)
    # img2_tr=img2_tr.unsqueeze(0).to(device)
    # img3_tr=img3_tr.unsqueeze(0).to(device)

    # predicted1=model(img1_tr)
    # predicted_class_1=torch.argmax(predicted1,dim=1)
    # predicted2=model(img2_tr)
    # predicted_class_2=torch.argmax(predicted2,dim=1)
    # predicted3=model(img3_tr)
    # predicted_class_3=torch.argmax(predicted3,dim=1)

    # print(predicted_class_1,predicted_class_2,predicted_class_3)
    # print(actions[1623],actions[752],actions[3248])
    # cv2.imshow("img1",img1)
    # cv2.waitKey(5000)
    # cv2.imshow("img2",img2)
    # cv2.waitKey(5000)
    # cv2.imshow("img3",img3)
    # cv2.waitKey(5000)
    

    





    