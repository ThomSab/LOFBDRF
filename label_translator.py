import numpy as np
import cv2
import pandas as pd
import os
import json
from PIL import Image

#as defined in the drss ieee competition:
color_dict = {(  0, 208,   0) :1,
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


def rgb_prediction_to_classification(rgb_prediction):
    if rgb_prediction.ndim==3:
        return np.array([color_dict[tuple(rgb_value)] for rgb_prediction_column in rgb_prediction for rgb_value in rgb_prediction_column]).reshape(rgb_prediction.shape[0],rgb_prediction.shape[1])
    return np.array([color_dict[tuple(rgb_value)] for rgb_value in rgb_prediction]) 
        
def perfomance(label_map,ground_truth):
    assert label_map.shape==ground_truth.shape, f"shapes not matching in performance calculation: \nlabel_map{label_map.shape} \nground_truth{ground_truth.shape}"
    overall_pxl = np.count_nonzero(label_map)
    correct_guesses = 0
    for column in range(label_map.shape[0]):
        for pxl in range(label_map.shape[1]):
            if label_map[column][pxl] == ground_truth[column][pxl] and ground_truth[column][pxl]!=0:
                correct_guesses += 1
    return correct_guesses/overall_pxl
                
def check_classes(array):
    class_set = set([ _ for col in array for _ in col])
    return list(class_set)

if __name__ == "__main__":
        
    label_folder_path = r"C:\Users\jasper\Documents\HTCV_local\Label_Maps"

    ground_truth = cv2.imread(r"C:\Users\jasper\Documents\HTCV_local\2018IEEE_Contest\Phase2\TrainingGT\2018_IEEE_GRSS_DFC_GT_TR.tif",0)
    


    label_image_paths = [
        [f"{label_folder_path}\\{directory}\\{file}"
            for file in os.listdir(label_folder_path+"\\"+directory)] 
                for directory in os.listdir(label_folder_path)]
                    #if "-med" not in directory and "-uni" not in directory] -med and -uni are valid labelmaps

    performance_dict = {_:0 for hyperfolder in label_image_paths for _ in hyperfolder}
    flat_label_image_paths = [_ for hyperfolder in label_image_paths for _ in hyperfolder]

    rgb_images = [cv2.imread(image) for directory in label_image_paths for image in directory ]
    """
    for label_image in rgb_images:
        classification_image = rgb_prediction_to_classification(label_image)
        
        out_img = Image.fromarray(label_image.astype(np.uint8))
    """
    for _ in range(len(rgb_images)):
        label_map = rgb_images[_]
        #test_img = rgb_prediction_to_classification(test_img)
        map_training_region = label_map[601:1202,596:2980]
        image_to_rescale = Image.fromarray(map_training_region.astype(np.uint8))
        image_rescaled = image_to_rescale.resize((4768,1202),Image.NEAREST)
        img_training_region = np.array(image_rescaled)
        img_training_region = rgb_prediction_to_classification(img_training_region)
        
        
        
        performance_dict[flat_label_image_paths[_]] = str(perfomance(img_training_region,ground_truth))
        print(flat_label_image_paths[_])
        print(check_classes(img_training_region))
        print(performance_dict[flat_label_image_paths[_]])
    
    
    with open(r"C:\Users\jasper\Documents\HTCV\LOFBDRF\performance_cleaned.json", "w+") as out_file:
        json.dump(performance_dict, out_file) 
