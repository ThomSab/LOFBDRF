import numpy as np
import cv2
import pandas as pd
import os
from PIL import Image

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
        self.rgb_prediction = None
        self.classification = None

def mock_create_grey_treesPredictions(RF,grey_prediction_vectors):
    #creates the treesPredictions vector as described as pseudocode in the paper
    grey_treesPredictions={}
    for idx,a_tree in enumerate(RF.estimators_):
        a_tree.prediction = grey_prediction_vectors[idx] #described as C(Tree(c_i),T) in the paper
        #input(type(a_tree.prediction))
        grey_treesPredictions[a_tree]=a_tree.prediction
    return grey_treesPredictions

def mock_create_rgb_treesPredictions(RF,rgb_prediction_vectors):
    for idx,a_tree in enumerate(RF.estimators_):
        a_tree.rgb_prediction = rgb_prediction_vectors[idx] 

def rgb_prediction_to_classification(rgb_prediction):
    return np.array([color_dict[tuple(rgb_value)] for rgb_value in rgb_prediction]) 

def mock_create_classifications(RF):
    for idx,a_tree in enumerate(RF.estimators_):
        print(f"Creating classification of tree {a_tree.name}")
        a_tree.classification = rgb_prediction_to_classification(a_tree.rgb_prediction)
        
def mock_majority_vote(mockforest):
    print("Voting...")
    ensemble_array=np.array([estimator.classification for estimator in mockforest.estimators_])
    return np.array([majority_classification(pixel) for pixel in ensemble_array.T])

def majority_classification(vector):
    return np.bincount(vector).argmax()

if __name__ == "__main__":
        
    label_folder_path = r"C:\Users\jasper\Documents\HTCV\MrsAdam\HTCV\Label_Maps"

    label_image_paths = [
        [f"{label_folder_path}\\{directory}\\{file}"
            for file in os.listdir(label_folder_path+"\\"+directory)] 
                for directory in os.listdir(label_folder_path)
                    if "-med" not in directory and "-uni" not in directory]


    rgb_images = [cv2.imread(image) for directory in label_image_paths for image in directory ]
    grey_images = [cv2.imread(image,0) for directory in label_image_paths for image in directory ]
    image_names = [image for directory in label_image_paths for image in directory ]
    #rgb_images = [cv2.imread(image)for directory in label_image_paths for image in directory ]


            

    grey_vector_images = [img.reshape(-1) for img in grey_images]
    #input(grey_vector_images[0][:100])

    
    rgb_vector_images = [img.reshape(-1,3) for img in rgb_images]
    #print(rgb_vector_images[0][:100])



    
    mockforest=MockForest(image_names)
    mock_create_grey_treesPredictions(mockforest,grey_vector_images)
    mock_create_rgb_treesPredictions(mockforest,rgb_vector_images)

    tree_predictions_dict = {tree:tree.prediction for tree in mockforest.estimators_}

    k = 50
    
    lof.RF_assign_dist(mockforest)
    lof.RF_assign_k_dist(mockforest,k)
    lof.RF_assign_N_k(mockforest,k)
    lof.RF_assign_lrd_k(mockforest,k)

    LOFs = lof.LOFs_from_treesPredictions(tree_predictions_dict,k)
    weights = lof.assemble_weigths(LOFs,None,None)

    mockforest.estimators_ = [tree for tree,weight in sorted(weights.items(), key=lambda item:item[1], reverse=True)][:k]

    print("Forest pruned")

    mock_create_classifications(mockforest)
    majority_vote= mock_majority_vote(mockforest)
    
    best_estimator = majority_vote
    best_estimator = np.reshape(best_estimator,(1202,4172))

    out_img = Image.fromarray(best_estimator.astype(np.uint8))
    out_img = out_img.resize((8344,2404),Image.NEAREST)
    """
    a prime example of why array coordinates and image coordinates dont mix well
    Because arrays are ordered by vectors and then rows, values are accessed by y and then x
    with images its the opposite so x and then y
    so the array has to reshaped as (y,x)
    then transformed to image
    and then resized as (x,y)
    """
    out_img.save(r"C:\Users\jasper\Documents\HTCV\MrsAdam\HTCV\estimator.tif" ,compression='raw')

    arr = np.array(out_img)
    class_set = set([ _ for col in arr for _ in col])
    print(list(class_set))


