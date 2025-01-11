import os
import cv2

from natsort import natsorted  
from utils.logger import  Logger
from utils.annotation_model import annotation
import pickle
import numpy as np

logger = Logger("dataloader","dataloader.log")


dataset_path="./dataset"
train_videos=[1,3,6,7,10,13,15,16,18,22,23,31,32,36,38,39,40,41,42,48,50,52,53,54]
validation_videos=[0,2,8,12,17,19,24,26,27,28,30,33,46,49,51]
test_videos=[4,5,9,11,14,20,21,25,29,34,35,37,43,44,45,47]


def save_dataset(X,y,image_file="images.npy",annotation_file="annotation.pkl"):
    
    np.save(image_file,np.array(X))
    
    with open(annotation_file, 'wb') as f :
        pickle.dump(y,f)
    logger.info("data saved successfully")

def load_data(image_file="images.npy",annotation_file="annotation.pkl"):
    X=np.load(image_file)
    logger.info("image loaded successfully")
    with open(annotation_file,'rb') as f:
        y=pickle.load(f) 

    return X,y


def load_annotations(annotation_path):

    annotations_per_clip=[]
    all_lines=[]
    with open(annotation_path,'r') as file:
        for line in file:
            all_lines.append(line)

    all_lines=natsorted(all_lines)
    for line in all_lines:
        cur_annotation=annotation(line)
        annotations_per_clip.append(cur_annotation)
    
    return annotations_per_clip

    

def show_video(clip_path,annot):
    
    images=os.listdir(clip_path)

    for image_name in images:
        image_path=os.path.join(clip_path,image_name)
        image=cv2.imread(image_path,1)
        image=cv2.resize(image,(1280,720))
        cv2.imshow(annot.action,image)
        cv2.waitKey(30)
    
    cv2.destroyAllWindows()



def show_image(image_path,annot):
    
    image=cv2.imread(image_path,1)

    logger.info(len(annot.players))
    for i in range(11):
        image=cv2.rectangle(image,(annot.players[i].x1,annot.players[i].y1),(annot.players[i].x1+annot.players[i].w,annot.players[i].y1+annot.players[i].h),(0, 0, 255),3)
    cv2.imshow("players annotated",image)
    cv2.waitKey(10)


    

def extract_players_images():    

    if not os.path.exists("players_dataset"):
        os.makedirs("players_dataset")
    else:
        logger.info("players_dataset already exists")
        return
    tournaments=natsorted(os.listdir(dataset_path)[:-1])
    X=[]
    y=[]
    
    global_index=0
    all_actions=[]
    players_images_paths=[]
    player_standing=0
    for tournament in tournaments:
        
        tournament_path=os.path.join(dataset_path,tournament)
        matches=natsorted(os.listdir(tournament_path))
        for match in matches:
            logger.info(f"reading for tournament {tournament} and match {match}")
            match_path=os.path.join(tournament_path,match)
            files = natsorted(os.listdir(match_path))
            clips, annotation = files[:-1], files[-1]
            annotation_per_match=load_annotations(os.path.join(match_path,annotation))
            for idx,clip in enumerate(clips):
                clip_path=os.path.join(match_path,clip)
                main_image_name=annotation_per_match[idx].main_image
                main_image_path=os.path.join(clip_path,main_image_name)
                image=cv2.imread(main_image_path,1)

                for p_idx,player in enumerate(annotation_per_match[idx].players):
                    player_image=image[player.y1:player.y1+player.h,player.x1:player.x1+player.w]
                    player_action=player.action
                    if player_action=="standing":
                        player_standing+=1
                    
                    
                    if player_standing%6==0 or player_action!="standing":
                        all_actions.append(player_action)
                        cv2.imwrite(f"./players_dataset/{global_index}_{p_idx}_{main_image_name}.jpg",player_image)
                        players_images_paths.append(f"./players_dataset/{global_index}_{p_idx}_{main_image_name}.jpg")
                global_index+=1
    with open("./players_dataset/actions.pkl", 'wb') as f:
        pickle.dump(all_actions, f)
    with open("./players_dataset/players_images_paths.pkl", 'wb') as f:
        pickle.dump(players_images_paths, f)
    logger.info("all actions saved successfully")
    return X,y
        



def get_dataset_first_time(all_frames=False):
    videos_train,videos_validation,videos_test=[],[],[]
    annotations_train,annotations_validation,annotations_test=[],[],[]
    

    
    for video in range(0,55):
        logger.info(f"reading for video {video}")
        video_path=os.path.join(dataset_path,str(video))
        files = natsorted(os.listdir(video_path))
        
        clips, annotation = files[:-1], files[-1]
        annotation_per_match=load_annotations(os.path.join(video_path,annotation))
        
        for idx,clip in enumerate(clips):
            clip_path=os.path.join(video_path,clip)
            
            # show_video(clip_path,annotation_per_match[idx])
            main_image_name=annotation_per_match[idx].main_image
            logger.info(f"clip_path {clip_path} main image name {main_image_name}")
            main_image_path=os.path.join(clip_path,main_image_name)
            images_to_be_appended=[]
            
            if all_frames and (video in train_videos): 
                images_in_clip=os.listdir(clip_path)
                for image_name in images_in_clip[18:23]:
                    image_path=os.path.join(clip_path,image_name)
                    if video in train_videos:
                        images_to_be_appended.append(image_path)
                    elif video in validation_videos:
                        images_to_be_appended.append(image_path)
                    else:
                        images_to_be_appended.append(image_path)
            else:
                images_to_be_appended.append(main_image_path)
            
            if video in train_videos:
                logger.info(f"main image path {main_image_path} annotation {annotation_per_match[idx].action}")
                videos_train.extend(images_to_be_appended)
                annotations_train.extend([annotation_per_match[idx]]*len(images_to_be_appended))
                # annotations_train.append(annotation_per_match[idx])
            elif video in validation_videos:
                videos_validation.extend(images_to_be_appended)
                annotations_validation.extend([annotation_per_match[idx]]*len(images_to_be_appended))
            else:
                videos_test.extend(images_to_be_appended)
                annotations_test.extend([annotation_per_match[idx]]*len(images_to_be_appended))
    
    logger.info("data extracted successfully")
    return videos_train,annotations_train,videos_validation,annotations_validation,videos_test,annotations_test
        


    


def get_data(all_frames=False):

    
    if os.path.exists("saved_pickles"):
        logger.info("we already saved the data before")
        train_videos,annotations_train=load_data(image_file="saved_pickles/train_images.npy",annotation_file="saved_pickles/train_annotation.pkl")
        videos_validation,annotations_validation=load_data(image_file="saved_pickles/validation_images.npy",annotation_file="saved_pickles/validation_annotation.pkl")
        videos_test,annotations_test=load_data(image_file="saved_pickles/test_images.npy",annotation_file="saved_pickles/test_annotation.pkl")
        
        logger.info("data retrieved")
    else:
        os.makedirs("saved_pickles")
        train_videos,annotations_train,videos_validation,annotations_validation,videos_test,annotations_test=get_dataset_first_time(all_frames)
        save_dataset(train_videos,annotations_train,image_file="saved_pickles/train_images.npy",annotation_file="saved_pickles/train_annotation.pkl")
        save_dataset(videos_validation,annotations_validation,image_file="saved_pickles/validation_images.npy",annotation_file="saved_pickles/validation_annotation.pkl")
        save_dataset(videos_test,annotations_test,image_file="saved_pickles/test_images.npy",annotation_file="saved_pickles/test_annotation.pkl")

    return train_videos,annotations_train,videos_validation,annotations_validation,videos_test,annotations_test



    
    


if __name__=="__main__":

    get_data()
