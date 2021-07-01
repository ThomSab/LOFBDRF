import numpy as np
import cv2
import pandas as pd
import os
from PIL import Image
import json
import random
from copy import deepcopy

import LOFB_DRF as lof


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

class MockForest:
    def __init__(self,tree_names):
        self.estimators_ = [MockTree(treename) for treename in tree_names]
        

class MockTree:
    
    def __init__(self,name):
        self.name = name
        self.prediction = None
        self.prediction_sample = None

def mock_create_treesPredictions(RF,rgb_prediction_vectors):
    for idx,a_tree in enumerate(RF.estimators_):
        a_tree.prediction = rgb_prediction_vectors[idx] 

def mock_create_prediction_samples(RF,prediction_samples):
    for idx,a_tree in enumerate(RF.estimators_):
        a_tree.prediction_sample = prediction_samples[idx] 

def rgb_prediction_to_classification(rgb_prediction):
    rgb_prediction_tuples = list(map(tuple,rgb_prediction))
    return np.array(list(map(color_dict.get,rgb_prediction_tuples))) 

def mock_create_classifications(RF):
    for idx,a_tree in enumerate(RF.estimators_):
        print(f"Creating classification of tree {a_tree.name}")
        a_tree.prediction = rgb_prediction_to_classification(a_tree.prediction)

def mock_create_sample_classifications(RF):
    for idx,a_tree in enumerate(RF.estimators_):
        print(f"Creating classification of tree {a_tree.name}")
        a_tree.prediction_sample = rgb_prediction_to_classification(a_tree.prediction_sample)

        
def mock_majority_vote(mockforest):
    print("Voting...")
    ensemble_array=np.array([estimator.prediction for estimator in mockforest.estimators_])
    return np.array([majority_classification(pixel) for pixel in ensemble_array.T])
    
def extract_sample_from_vector(vector,sample_idx_list=None):
    if sample_idx_list is None:
        return vector
    else:
        return np.array([vector[_] for _ in sample_idx_list])
        

def majority_classification(vector):
    return np.bincount(vector).argmax()

def estimator_to_upload_image(estimator):
    estimator = np.reshape(estimator,(1202,4172))
    out_img = Image.fromarray(estimator.astype(np.uint8))
    out_img = out_img.resize((8344,2404),Image.NEAREST)
    """
    a prime example of why array coordinates and image coordinates dont mix well
    Because arrays are ordered by vectors and then rows, values are accessed by y and then x
    with images its the opposite so x and then y
    so the array has to reshaped as (y,x)
    then transformed to image
    and then resized as (x,y)
    """
    return out_img

def img_to_classes_list(estimator_img):
    arr = np.array(estimator_img)
    class_set = set([ _ for col in arr for _ in col])
    return list(class_set)
    
    

    
if __name__ == "__main__":
        
    sample_idx_list=None
    
    
    label_folder_path = r"C:\Users\jasper\Documents\HTCV_local\Label_Maps"
    
    with open(r"C:\Users\jasper\Documents\HTCV\LOFBDRF\performance.json") as score_file:
        score_dict = json.load(score_file)
    
    label_image_paths = [
        [f"{label_folder_path}\\{directory}\\{file}"
            for file in os.listdir(label_folder_path+"\\"+directory)] 
                for directory in os.listdir(label_folder_path)]
                    #if "-med" not in directory and "-uni" not in directory]
    image_names = [image for directory in label_image_paths for image in directory ]

                    
    rgb_image_dict = {image : cv2.imread(image) for image in image_names}
    rgb_vector_images = [img.reshape(-1,3) for img in rgb_image_dict.values()]

    base_mockforest=MockForest(image_names) #original Tree to be copied
    mock_create_treesPredictions(base_mockforest,rgb_vector_images)
    for tree in mockforest.estimators_:
            tree.score_attr = np.float(score_dict[tree.name])
    mock_create_classifications(base_mockforest)#this takes a while

    
    for k in [5,9,10,15,20,25,30,35,40,45,50,50,70,90,10,120,140]:
        for sample_size in [100,10000,100000]:
            for run in range(3):
                sample_idx_list = np.random.choice(range(len(rgb_vector_images[0])),sample_size,replace=False)
                rgb_vector_image_samples = [extract_sample_from_vector(vector_image,sample_idx_list) for vector_image in rgb_vector_images]

                mockforest = deepcopy(base_mockforest)
                
                mock_create_prediction_samples(mockforest,rgb_vector_image_samples) 

                mockforest.estimators_.sort(key= lambda x:x.score_attr,reverse=True)
                mock_create_sample_classifications(mockforest)

                tree_predictions_dict = {tree:tree.prediction for tree in mockforest.estimators_}
                
                lof.RF_assign_dist(mockforest)
                lof.RF_assign_k_dist(mockforest,k)
                lof.RF_assign_N_k(mockforest,k)
                lof.RF_assign_lrd_k(mockforest,k)

                LOFs = lof.LOFs_from_treesPredictions(tree_predictions_dict,k)
                weights = lof.LOFs_and_score_to_weights(LOFs)

                mockforest.estimators_ = [tree for tree,weight in sorted(weights.items(), key=lambda item:item[1], reverse=True)][:k]

                print("Forest pruned")

                majority_vote= mock_majority_vote(mockforest)
                
                best_estimator = majority_vote
                estimator_img = estimator_to_upload_image(best_estimator)
                estimator_img.save(fr"C:\Users\jasper\Documents\HTCV_local\estimator_s{sample_size}_k{k}_run{run}.tif" ,compression='raw')

                best_estimator_visible = np.array([_*255/20 for _ in best_estimator])
                estimator_img_visible = estimator_to_upload_image(best_estimator_visible)
                estimator_img_visible.save(fr"C:\Users\jasper\Documents\HTCV_local\estimator_visible_s{sample_size}_k{k}_run{run}.tif" ,compression='raw')
                
                print(img_to_classes_list(estimator_img_visible))

