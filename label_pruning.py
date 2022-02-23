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
              
platt_c_dict = { ( 255, 255, 255 ):0,   #No labels 
                 ( 255, 0,   0   ):1,   #urban   
                 ( 0,   128, 0   ):2,   #forest   
                 ( 255, 0,   255 ):3,   #road    
                 ( 255, 255, 0   ):4,   #field    
                 ( 0,   0,   255 ):5 }  #water    

opr_c_dict = {(0,   0,   0  ):0,   #No labels 
              (255, 0,   0  ):1,   #urban     
              (0,   128, 0  ):2,   #forest    
              (0,   0,   255):3,   #road      
              (255, 255, 0  ):4,   #field     
              (0,   255, 0  ):5 }  #grassland 

class MockForest:
    def __init__(self,tree_names):
        self.estimators_ = [MockTree(treename) for treename in tree_names]
        

class MockTree:
    
    def __init__(self,name):
        self.name = name
        self.prediction = None
        self.prediction_sample = None

class PruningInformation:
    def __init__(self):
        self.k = 0
        self.sample_size = 0
        self.run = 0
        self.base_mockforest =None
        

def mock_create_treesPredictions(RF,classification_vector_dict):
    for a_tree in RF.estimators_:
        a_tree.prediction = classification_vector_dict[a_tree.name] 

def mock_create_prediction_samples(RF,sample_idx_list):
    print("\t\tcreating sample classifications...")
    assert RF.estimators_[0].prediction[0] in range(21), f"tree prediction has not been transformed from rgb to classification: {RF.estimators_[0].prediction[0]}"
    for a_tree in RF.estimators_:
        a_tree.prediction_sample = np.array([ a_tree.prediction[idx] for idx in sample_idx_list])


        
def mock_majority_vote(mockforest):
    print("\t\tVoting...")
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
    """
    a prime example of why array coordinates and image coordinates dont mix well
    Because arrays are ordered by vectors and then rows, values are accessed by y and then x
    with images its the opposite so x and then y
    so the array has to reshaped as (y,x)
    then transformed to image
    """
    if estimator.shape == (9229600,): #for oph
        estimator = np.reshape(estimator,(6640,1390))
        out_img = Image.fromarray(estimator.astype(np.uint8))
        #out_img = out_img.resize((x,y),Image.NEAREST) --> used to be necessary

    if estimator.shape == (120629776,):
        estimator = np.reshape(estimator,(11698,10312))
        out_img = Image.fromarray(estimator.astype(np.uint8))
        
    return out_img

def img_to_classes_list(estimator_img):
    arr = np.array(estimator_img)
    class_set = set([ _ for col in arr for _ in col])
    return list(class_set)

def generate_test_forest(pruning_information):
    sample_idx_list = np.random.choice(range(len(base_mockforest.estimators_[0].prediction)),pruning_information.sample_size,replace=False)
    mockforest = deepcopy(pruning_information.base_mockforest)
    
    mock_create_prediction_samples(mockforest,sample_idx_list) 

    mockforest.estimators_.sort(key= lambda x:x.score_attr,reverse=True)
    
    return mockforest

def calculate_lofs(mockforest,k):
    print("\t\tcalculating lofs")
    tree_predictions_dict = {tree:tree.prediction for tree in mockforest.estimators_}
    lof.RF_assign_dist(mockforest)
    lof.RF_assign_k_dist(mockforest,k)
    lof.RF_assign_N_k(mockforest,k)
    lof.RF_assign_lrd_k(mockforest,k)
    return lof.LOFs_from_treesPredictions(tree_predictions_dict,k)
    
def prune(pruning_information, print_leftovers = False):
    mockforest = generate_test_forest(pruning_information)
    
    LOFs = calculate_lofs(mockforest,pruning_information.k)
    weights = lof.LOFs_and_score_to_weights(LOFs)

    mockforest.estimators_ = [tree for tree,weight in sorted(weights.items(), key=lambda item:item[1], reverse=True)][:pruning_information.k]

    print("\t\tForest pruned")

    if print_leftovers:
        for estimator in mockforest.estimators_:
            print(f"\t\t{estimator.name}")
            
    return mockforest

                

def save(majority_vote,pruning_information):
    print("\t\tsaving...")
    best_estimator = majority_vote
    estimator_img = estimator_to_upload_image(best_estimator)
    estimator_img.save(fr"C:\Users\jasper\Documents\HTCV_local\{location}_pruned\{location}_estimator_s{pruning_information.sample_size}_k{pruning_information.k}_run{pruning_information.run}.tif" ,compression='raw')

    best_estimator_visible = np.array([_*255/20 for _ in best_estimator])
    estimator_img_visible = estimator_to_upload_image(best_estimator_visible)
    estimator_img_visible.save(fr"C:\Users\jasper\Documents\HTCV_local\{location}_pruned\{location}_estimator_visible_s{pruning_information.sample_size}_k{pruning_information.k}_run{pruning_information.run}.tif" ,compression='raw')

    print("\t\tsaved output images to drive")
    #print(img_to_classes_list(estimator_img_visible))
    
def prune_and_save(pruning_information):
    pruned_mockforest = prune(pruning_information)
    majority_vote = mock_majority_vote(pruned_mockforest)
    save(majority_vote,pruning_information)
    
    save_path=fr"C:\Users\jasper\Documents\HTCV_local\{location}_pruned"
    with open(save_path + f'\\k{pruning_information.k}_s{pruning_information.sample_size}_r{pruning_information.run}.json', 'w') as f:
        forest_members = [tree.name for tree in pruned_mockforest.estimators_]
        json.dump(forest_members, f)


def run_experiment(k_list,sample_list,run_list):
    for k in k_list:#[5, 28, 56, 84, 112, 140]:
        assert k>0, f"k must be bigger than zero! Is {k}"
        print(f"k is {k}")
        for sample_size in sample_list:#[100,10000,100000]:
            print(f"\tsample size is {sample_size}")
            for run in run_list:
                pruning_information.k = k
                pruning_information.sample_size = sample_size
                pruning_information.run = run
                print(f"\t\tgoing through run {run}")
                prune_and_save(pruning_information)
                
                

if __name__ == "__main__":
        
    sample_idx_list=None
    
    location="oph"
    label_folder_path = fr"C:\Users\jasper\Documents\HTCV_local\{location}_Label_Maps_Grey"
    
    #old path r"C:\Users\jasper\Documents\LOFBDRF\performance_samir.json"
    with open(fr"C:\Users\jasper\Documents\HTCV_local\accs\{location}_acc.json") as score_file:
        score_dict = json.load(score_file)

    top_10_folders = ["rgb-h15-s10000-t10",
                      "rgb-h30-s10000-t10",
                      "rgb-h30-s10000-t10-med",
                      "rgb-h30-s10000-t10-uni",
                      "hsi-h15-s10000-t10",
                      "rgb-h5-s10000-t10",
                      "hsi-h30-s10000-t10",
                      "rgb-h5-s100-t10",
                      "hsi-h30-s10000-t10",
                      "rgb-h30-s100-t10"]
                      
    top_1_folder = ["rgb-h15-s10000-t10"]

    classification_image_dict =   {directory+"-tree"+image.split('tree-')[-1][0] : cv2.imread(label_folder_path+"\\"+directory+"\\"+image, cv2.IMREAD_GRAYSCALE)
                        for directory in os.listdir(label_folder_path) 
                            for image in os.listdir(label_folder_path+"\\"+directory)
                                if "visible" not in image and "_est-0" not in image }

    
    
    classifiation_vector_dict = {tree_name:classification.reshape(-1) for tree_name,classification in classification_image_dict.items()}

    base_mockforest=MockForest(classification_image_dict.keys()) #original Tree to be copied
    mock_create_treesPredictions(base_mockforest,classifiation_vector_dict)
    
    for tree in base_mockforest.estimators_:
        tree.score_attr = np.float(score_dict[tree.name])
        
    
    pruning_information = PruningInformation()
    pruning_information.base_mockforest = base_mockforest
    
    run_experiment(list(np.arange(28,71,1)),[100000],range(3))
    #run_experiment([7],[100000],range(1))
