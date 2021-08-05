import os
from sys import path

from numpy.lib.arraypad import pad
from helper import create_data_clusters, parse_xml
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2


DATA_PATH = os.path.join(os.getcwd(),"data")
MALIGNANT_PATH = os.path.join(DATA_PATH,"malignant")
BENIGN_PATH = os.path.join(DATA_PATH,"bening")
UNDEFINED_PATH = os.path.join(DATA_PATH,"undefined")

#PREPROCESSED PATHS
PROCESSED_DATA_PATH = os.path.join(os.getcwd(),"processed_data")
PROCESSED_MALIGNANT_PATH = os.path.join(PROCESSED_DATA_PATH,"malignant")
PROCESSED_BENIGN_PATH = os.path.join(PROCESSED_DATA_PATH,"bening")
PROCESSED_UNDEFINED_PATH = os.path.join(PROCESSED_DATA_PATH,"undefined")

BENING_TRIADS = ['2', '3']
MALIGNANT_TRIADS = ['5', '4a', '4c','4b']


def create_folder():
    #Creates relative folders in data directory
    #Raw Data Folders
    if not os.path.exists(MALIGNANT_PATH): os.mkdir(MALIGNANT_PATH)
    if not os.path.exists(BENIGN_PATH): os.mkdir(BENIGN_PATH) 
    if not os.path.exists(UNDEFINED_PATH): os.mkdir(UNDEFINED_PATH) 

    #Processed Data FOlders
    if not os.path.exists(PROCESSED_DATA_PATH): os.mkdir(PROCESSED_DATA_PATH)
    if not os.path.exists(PROCESSED_MALIGNANT_PATH): os.mkdir(PROCESSED_MALIGNANT_PATH)
    if not os.path.exists(PROCESSED_BENIGN_PATH): os.mkdir(PROCESSED_BENIGN_PATH) 
    if not os.path.exists(PROCESSED_UNDEFINED_PATH): os.mkdir(PROCESSED_UNDEFINED_PATH) 


def move_image(source, img_class):
    #Moves the given image path to corresponding class

    if img_class == "BENIGN":
        img_name = os.path.split(source)[1]
        destination = os.path.join(BENIGN_PATH,img_name)
        if not os.path.exists(destination):
            os.replace(source,destination)
    if img_class == "MALIGNANT":
        img_name = os.path.split(source)[1]
        destination = os.path.join(MALIGNANT_PATH,img_name)
        if not os.path.exists(destination):
            os.replace(source,destination)

    if img_class == "UNDEFINED":
        img_name = os.path.split(source)[1]
        destination = os.path.join(UNDEFINED_PATH,img_name)
        if not os.path.exists(destination):
            os.replace(source,destination)


def split_to_folders():
    #Get the data
    data_paths = create_data_clusters()
    #Split the data into corresponding classes -- malignant, benign, undefined

    for idx, data, in data_paths.items():
        if data["annot"+str(idx)] != [] and data["img"+str(idx)] != []:

            try:
                xml_file_path = data["annot"+str(idx)][0]
                data_class = parse_xml(xml_file_path)

                if data_class in BENING_TRIADS:
                    img_source = data["img"+str(idx)]
                    for source in img_source:
                        move_image(source, "BENIGN")

                elif data_class in MALIGNANT_TRIADS:
                    img_source = data["img"+str(idx)]
                    for source in img_source:
                        move_image(source,"MALIGNANT")

                else:
                    img_source = data["img"+str(idx)]
                    for source in img_source:
                        move_image(source,"UNDEFINED")

            except IndexError:
                pass

def save_processed_images():
    os.chdir(PROCESSED_MALIGNANT_PATH)
    for img_name in os.listdir(MALIGNANT_PATH):
        malig_img =cv2.imread(os.path.join(MALIGNANT_PATH,img_name))
        cropped_img = malig_img[6:308,85:470]
        cv2.imwrite(img_name,cropped_img)
    
    os.chdir(PROCESSED_BENIGN_PATH)
    for img_name in os.listdir(BENIGN_PATH):
        malig_img =cv2.imread(os.path.join(BENIGN_PATH,img_name))
        cropped_img = malig_img[6:308,85:470]
        cv2.imwrite(img_name,cropped_img)
        


def main():
    #Create split into classes - Benign or Malignant
    create_folder()
    split_to_folders()
    save_processed_images()
    



    
    
    
    #img = random.choice(os.listdir(MALIGNANT_PATH))
    #img = Image.open(os.path.join(MALIGNANT_PATH,img))
    #img = cv2.imread(os.path.join(MALIGNANT_PATH,img))
    #cropped_img = img[6:308,85:470]
    #cv2.imshow("img", cropped_img)
    #cv2.imshow("img2", img)
    #cv2.waitKey(0)

    
    


if __name__ == "__main__":
    main()

     