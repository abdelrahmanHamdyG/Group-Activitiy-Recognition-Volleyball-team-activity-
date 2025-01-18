from dataloading.dataloader import get_players_dataset,save_checkpoint
from modeling.player_level_classification.player_level_classification_model import PlayerDataset,load_players_model
from torch.utils.data import random_split
from torchvision import transforms as T
from collections import Counter
import torch
import torch.nn as nn
from utils.logger import Logger
logger=Logger("player_level_classification_model","modeling/player_level_classification/player_level_classification.log")



train_transform = T.transforms.Compose([
    T.transforms.Resize((256, 256)),  # Resize to a larger size for more cropping options
    T.transforms.RandomCrop((224, 224)),  # Randomly crop to 224x224
    T.transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
    T.RandomApply([
        T.RandomChoice([
            T.GaussianBlur(kernel_size=3),
            T.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.3, hue=0.1),
            T.RandomRotation(degrees=30),  # Randomly rotate the image
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # Random affine transformations
        ])
    ], p=0.5),  
    T.RandomGrayscale(p=0.2),  # Randomly convert to grayscale
    T.transforms.ToTensor(),
    T.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = T.transforms.Compose([
    T.transforms.Resize((224, 224)),
    
    T.transforms.ToTensor(),
    T.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train():
    players_images,actions=get_players_dataset()
    dataset = PlayerDataset(players_images, actions, transform=None)  # Create a dataset without splitting first

    
    train_size = int(0.9 * len(dataset))  
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size],generator=torch.Generator().manual_seed(42))

    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform



    train_labels = [actions[i] for i in train_dataset.indices]
    class_counts = Counter(train_labels)
    total_samples = sum(class_counts.values())
    logger.info(f"Class Counts: {class_counts}")
    alpha = 0.15 
    class_weights = torch.tensor(
        [(total_samples / class_counts.get(cls, 1)) ** alpha for cls in class_counts.keys()],
        dtype=torch.float32
    ).to('cuda' if torch.cuda.is_available() else 'cpu')


    model,optimizer,start_epoch=load_players_model(best_model=True,new_lr=0.00001,weight_decay=0.01)
    train_criterion=nn.CrossEntropyLoss(weight=class_weights)
    test_criterion=nn.CrossEntropyLoss()
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.65)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=False)
    

    num_epochs=15
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(start_epoch+1,num_epochs):
        model.train()
        train_loss=0
        train_correct=0
        logger.info(f"Training: Epoch {epoch} learning rate {optimizer.param_groups[0]['lr']}")
        for i,(images,labels) in enumerate(train_loader):
            if i%40==0 and i!=0:
                logger.info(f" Iteration {i}/{len(train_loader)} Loss {train_loss/(i)} Accuracy {train_correct/(i*64)}")
            images=images.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=train_criterion(outputs,labels)
            predicted_class=torch.argmax(outputs,dim=1)
            train_correct+=(predicted_class==labels).sum().item()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
        logger.info(f"Training Epoch {epoch} : Loss {train_loss/len(train_loader)} Accuracy {train_correct/len(train_dataset)}")    
            
            
        model.eval()
        
        with torch.no_grad():
            logger.info(f"Validation Epoch {epoch} :")
            test_loss=0
            test_correct=0
            total=0
            for idx,(images,labels) in enumerate(test_loader):
                if idx%10==0 and idx!=0:
                    logger.info(f"Validation Iteration {idx}/{len(test_loader)} Loss {test_loss/(idx)} Accuracy {test_correct/(idx*64)}")
                images=images.to(device)
                labels=labels.to(device)
                outputs=model(images)
                _,predicted=torch.max(outputs.data,1)
                total+=labels.size(0)
                loss=test_criterion(outputs,labels)
                test_loss+=loss.item()
                test_correct+=(predicted==labels).sum().item()
            
            logger.info(f"Testing Epoch {epoch} : Loss {test_loss/len(test_loader)} Accuracy {test_correct/total}")
        scheduler.step()
        save_checkpoint(epoch,model.state_dict(),optimizer.state_dict(),train_loss/len(train_loader),test_loss/len(test_loader),test_correct/total,"models_trained/player_level_classification/checkpoints","models_trained/player_level_classification/best_model.pth")
    


    


  



if __name__=="__main__":
    train()