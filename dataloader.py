import os
import cv2
dataset_path="./dataset"
from natsort import natsorted  
from logger import  Logger
from annotation_model import annotation
import pickle
import numpy as np

logger = Logger("dataloader")

def save_data(X,y,image_file="images.npy",annotation_file="annotation.pkl"):
    
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

    

def show_video(clip_path):
    print(clip_path)
    images=os.listdir(clip_path)

    for image_name in images:
        image_path=os.path.join(clip_path,image_name)
        image=cv2.imread(image_path,1)
        image=cv2.resize(image,(1280,720))
        cv2.imshow("video",image)
        cv2.waitKey(30)
    

        
        



def load_images():
    tournaments=natsorted(os.listdir(dataset_path)[:-1])
    X=[]
    y=[]
    all_annotations=[]
    for tournament in tournaments:
        
        tournament_path=os.path.join(dataset_path,tournament)
        matches=natsorted(os.listdir(tournament_path))
        for match in matches:
            logger.info(f"reading for tournament {tournament} and match {match}")
            match_path=os.path.join(tournament_path,match)
            files = natsorted(os.listdir(match_path))
            clips, annotation = files[:-1], files[-1]
            annotation_per_match=load_annotations(os.path.join(match_path,annotation))
            all_annotations.extend(annotation_per_match)
            for idx,clip in enumerate(clips):
                clip_path=os.path.join(match_path,clip)
                main_image_name=annotation_per_match[idx].main_image
                main_image_path=os.path.join(clip_path,main_image_name)
                main_image=cv2.imread(main_image_path,1)
                if main_image.shape[0]==1080:
                    main_image=cv2.resize(main_image,(1280,720))
                
                X.append(main_image)
                
                y.append(annotation_per_match[idx])
        
    return X,y
        


def main():

    images_file="images.npy"
    annotation_file="annotation.pkl"
    if os.path.exists(images_file) and os.path.exists(annotation_file):
        logger.info("we already saved the data before")
        main_images,annot=load_data()
        logger.info("data retrieved")
    else:
        
        main_images,annot=load_images()
        save_data(main_images,annot)
    
    

    print(len(main_images))
    print(len(annot))


if __name__=="__main__":
    main()



