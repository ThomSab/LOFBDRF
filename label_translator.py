import numpy as np
import cv2
import pandas as pd
import os
import json
from PIL import Image

#as defined in the drss ieee competition:
old_color_dict = {(  0, 208,   0) :1,
              (  0, 255, 123) :2,
              ( 96, 156,  49) :3,
              (  0, 143,   0) :4,
              (  0,  76,   0) :5,
              ( 47,  73, 146) :6,
              (242, 242,   0) :7,
              (255, 255, 255) :8,
              (216, 191, 216) :9,
              (  0,   0, 255) :10,
              (163, 172, 183) :11,
              (119, 104, 119) :12,
              (  0,   0, 167) :13,
              (  0,   0,  79) :14,
              ( 16, 155, 229) :15,
              (  0, 255, 255) :16,
              (  0,  95, 173) :17,
              (193,   0, 193) :18,
              (244,   0,   0) :19,
              (223, 199, 180) :20}


platt_c_dict = { ( 255, 255, 255 ):0,   #No labels 
                 ( 0  , 0,   255 ):1,   #urban   
                 ( 0,   128, 0   ):2,   #forest   
                 ( 255, 0,   255 ):3,   #road    
                 ( 0  , 255, 255 ):4,   #field    
                 ( 255,   0, 0   ):5 }  #water    

oph_c_dict = {(0,   0,   0  ):0,   #No labels 
              (0, 0,   255  ):1,   #urban     
              (0,   128, 0  ):2,   #forest    
              (255,   0, 0  ):3,   #road      
              (0, 255, 255  ):4,   #field     
              (0,   255, 0  ):5 }  #grassland 
              
              #(0, 255, 255),
              #(0, 255, 0  ),
              #(0, 128, 0  ),
              #(255, 0, 0  ),
              #(0,   0, 255)

color_dict = platt_c_dict

def rgb_prediction_to_classification(rgb_prediction):
    if rgb_prediction.ndim==3:
        return np.array([color_dict[tuple(rgb_value)] for rgb_prediction_column in rgb_prediction for rgb_value in rgb_prediction_column]).reshape(rgb_prediction.shape[0],rgb_prediction.shape[1])
    return np.array([color_dict[tuple(rgb_value)] for rgb_value in rgb_prediction]) 
        
def perfomance(label_map,ground_truth,trainingSample):
    assert label_map.shape==ground_truth.shape, f"shapes not matching in performance calculation: \nlabel_map{label_map.shape} \nground_truth{ground_truth.shape}"
    overall_pxl = np.count_nonzero(ground_truth)
    correct_guesses = 0
    for column in range(label_map.shape[0]):
        for pxl in range(label_map.shape[1]):
            if label_map[column][pxl] == ground_truth[column][pxl] and is_nonzero(column,pxl,ground_truth):
                if not is_training_sample(column,pxl,trainingSample):
                    correct_guesses += 1
                else:
                    overall_pxl-=1
                
    return correct_guesses/overall_pxl

def is_nonzero(x,y,ground_truth):
    return ground_truth[x][y]!=0

def is_training_sample(x,y,trainingSample):
    return trainingSample[x][y]==76#the greyscale value for the training pixels color ie red

def check_classes(array):
    class_set = set([ _ for col in array for _ in col])
    return list(class_set)

def cut_training_region_from_lm(label_map):
    assert (label_map.shape[0]==1202 and label_map.shape[1]==4172), "label_map is crooked from the start"
    map_training_region = label_map[601:1202,596:2980]
    image_to_rescale = Image.fromarray(map_training_region.astype(np.uint8))
    image_rescaled = image_to_rescale.resize((4768,1202),Image.NEAREST)
    return image_rescaled
    
    return training_region

def save_map(label_map,directory,file_name,location):
    print(f"saving {directory} // {file_name}...")
    label_map.save(fr"C:\Users\jasper\Documents\HTCV_local\{location}_Label_Maps_Grey\{directory}\{file_name}" ,compression='raw')
    

if __name__ == "__main__":
        
    #label_folder_path = r"C:\Users\jasper\Documents\HTCV_local\Label_Maps"
    #old labelmaps were in label maps but now its oph and platt
    
    label_folder_path = r"C:\Users\jasper\Documents\HTCV_local"
    
    location = "platt" # "platt"
    
    label_folder_path=label_folder_path+"\\"+location 
    
    
    for memory_idx in range(len(os.listdir(label_folder_path))-1):
    
    
        label_image_paths = [
            [f"{label_folder_path}\\{directory}\\{file}"
                for file in os.listdir(label_folder_path+"\\"+directory)
                    if "Samples" not in file]
                    for directory in os.listdir(label_folder_path)[memory_idx:memory_idx+1]
                        if directory not in ["label_inter.png","reference.png"]]
        
        flat_label_image_paths = [_ for hyperfolder in label_image_paths for _ in hyperfolder]
        rgb_images = [cv2.imread(image) for directory in label_image_paths for image in directory ]

        for idx,full_file_name in enumerate(flat_label_image_paths):

            print(full_file_name)
            rgb_classification = rgb_prediction_to_classification(rgb_images[idx])
            directory,file_name = full_file_name.split("\\")[-2],full_file_name.split("\\")[-1]
            try:
                os.mkdir(fr"C:\Users\jasper\Documents\HTCV_local\{location}_Label_Maps_Grey\{directory}")
            except FileExistsError:
                print("Directory Exists.")

            classification_image = Image.fromarray(rgb_classification.astype(np.uint8))
            test_image = Image.fromarray(np.array([_*255/len(color_dict) for _ in rgb_classification]).astype(np.uint8))
            save_map(test_image,directory,"visible_"+file_name.replace(".png",".tif"),location)
            save_map(classification_image,directory,file_name.replace(".png",".tif"),location)
        
