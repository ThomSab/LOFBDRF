import pandas as pd
import visualize
from numpy.linalg import norm
import numpy as np
import copy

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits




#k is constant and also the amount of trees remaining in the pruned forest
#even though i can see no direct connection between the two

    
def RF_assign_dist(RF):
    #calculates all distances between all vectors
    #and assigns the distances attribute to all trees 
    #st tree.distances is a dictionary with tree.distances[b_tree] gives the distance to b_tree
    #print("Calculating distances...")
    assert hasattr(RF.estimators_[0],"prediction"), "Random Tree has not been trained yet or the random Trees have not yet been assigned a prediction attribute."
    for a_tree in RF.estimators_:
        a_tree.distances = {b_tree: np.linalg.norm(a_tree.prediction_sample - b_tree.prediction_sample) for b_tree in RF.estimators_}
    
def k_dist(a_tree,k):
    #the k-distance of a, in this case a is the prediction made by a random tree over the test data
    #so in this case the paper specifies the prediction vector a as the input of the k-distance function
    #but it really doesnt make a difference bc a == a_tree.If so, don't I require a ground truth to determine the quality each trees output?
    distances  = [distance for b_tree,distance in sorted(a_tree.distances.items(), key=lambda item:item[1])]
    return distances[k] #very intuitive

def RF_assign_k_dist(RF,k):
    #print("Calculating k_distances...")
    for a_tree in RF.estimators_:
        a_tree.k_dist = k_dist(a_tree,k)

def rd_k(a_tree,b_tree,k):
    #the reachablility distance between to points a and b
    return max([b_tree.k_dist, a_tree.distances[b_tree]])

def N_k(a_tree,k):
    #the k nearest neighbors of a_tree
    #again the paper specifies a as input but here as well:
    #a == a_tree.prediction_sample
    
    """
    It is important to specify that N_k is not equal to k 
    because two points may be in equal distance to a_tree
    in that case N_k > k
    """
    within_k_dist  = [b_tree for b_tree,distance in sorted(a_tree.distances.items(), key=lambda item:item[1]) if distance <= a_tree.k_dist]
    
    return within_k_dist

def RF_assign_N_k(RF,k):
    #assigns the set of N_k nearest neighbours to each tree
    #print("Calculating N_k nearest neighbours...")
    for a_tree in RF.estimators_:
        a_tree.N_k = N_k(a_tree,k)

def lrd_k(a_tree,k):
    #the local reachablility density of a_tree
    return sum( [rd_k(a_tree,b_tree,k) for b_tree in a_tree.N_k] ) / len(a_tree.N_k)

def RF_assign_lrd_k(RF,k):
    #assigns a local reachablility density to each tree in the random forest
    #print("Calculating local reachablility densities...")
    for a_tree in RF.estimators_:
        a_tree.lrd_k = lrd_k(a_tree,k)

def LOF_k(a_tree,k):
    #the local outlier factor of a_tree 
    return sum([(b_tree.lrd_k/a_tree.lrd_k) for b_tree in a_tree.N_k])/ len(a_tree.N_k)

def create_treesPredictions(RF,x_test,y_test):
    #creates the treesPredictions vector as described as pseudocode in the paper
    treesPredictions={}
    for a_tree in RF:
        a_tree.prediction_sample = a_tree.predict(x_test) #described as C(Tree(c_i),T) in the paper
        #input(type(a_tree.prediction_sample))
        treesPredictions[a_tree]=a_tree.predict(x_test)
    return treesPredictions

def LOFs_from_treesPredictions(treesPredictions,k):
    #assigns an LOF to every tree in the ensemble 
    #and returns it as a dictionary
    #print("Calculating LOFs...")
    return {a_tree:LOF_k(a_tree,k) for a_tree,prediction in treesPredictions.items()}

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def normalize_LOF_dict(LOF_dict):
    #normalizes all the values in the LOF dict
    #st that they represent the probablility of each tree beeing an outlier
    #the paper is referencing a source but does not specify the method of normalization
    
    return {tree: LOF/sum(LOF_dict.values()) for tree,LOF in LOF_dict.items()} #for the time beeing ill just use a regular sigmoidal function

def tree_weight(tree,tree_LOF):
    #calculates the weight of the tree according to the paper
    #why they had to call it "weight" is beyond me
    if tree.score_attr==None:
        return tree_LOF
    return tree.score_attr * tree_LOF
    

def calculate_score(tree,x_test,y_test):
    tree.score_attr=tree.score(x_test,y_test)

def LOFs_and_score_to_weights(normalized_LOF_dict):
    #assigns a weight to each tree in the ensemble
    #and returns it as a dictionary
    #input(f"y_test is {y_test}")
    assert max(normalized_LOF_dict.values()), "Local Outlier Factors not normalized"
    weights = {tree:tree_weight(tree,LOF) for tree,LOF in normalized_LOF_dict.items() }
    return weights

#@cache 
def assemble_LOFs(RF,k,x_test,y_test):
    #print("assembling LOFs...")
    treesPredictions = create_treesPredictions(RF,x_test,y_test)
    RF_assign_dist(RF)
    RF_assign_k_dist(RF,k)
    RF_assign_N_k(RF,k)
    RF_assign_lrd_k(RF,k)
    
    LOFs = LOFs_from_treesPredictions(treesPredictions,k)
    return LOFs

def assemble_weigths(LOFs,x_test,y_test):
    #print("assembling weights...")
    normalized_LOF_dict = normalize_LOF_dict(LOFs)
    weights = weights_from_LOFs(normalized_LOF_dict,x_test,y_test)
    return weights

def RF_accuracies(RF,x_test,y_test):
    avg_tree_acc = np.mean([tree.score(x_test,y_test) for tree in RF.estimators_]) #the 256 th tree's score over the oob data set   
    ensemble_acc = RF.score(x_test,y_test) #out of bag accuracy of the ensemble
    return avg_tree_acc,ensemble_acc 

def assemble_LOFB_DRF(RF,k,x_test,y_test):

    LOFs = assemble_LOFs(RF,k,x_test,y_test)
    weights = assemble_weigths(LOFs,x_test,y_test)
    
    RF.estimators_ = [tree for tree,weight in sorted(weights.items(), key=lambda item:item[1], reverse=True)][:k]
    
    return LOFs,weights

if __name__ == "__main__":

    digits = load_digits() # load dataset
    x_train,x_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.34)

    """
    seperate the data into training and test data (or sometimes out-of-bag and in-bag samples)
    two and one thirds respectively as is specified in the paper on LOFB-DRF
    According to the paper this is very close to the ratio that that would result from bootstrapping
    with bootstrapping resulting in ~ 36% testdata to 64% trainingdata
    """
    
    
    
    RF = RandomForestClassifier(bootstrap=True,oob_score=True,n_estimators=100)
    RF.fit(x_train,y_train) #train the forest on the data
    
    avg_tree_acc,ensemble_acc = RF_accuracies(RF,x_test,y_test)
    #print(f"{avg_tree_acc} average accuracy over all trees")
    #print(f"{ensemble_acc} ensemble accuracy")

    for tree in RF:
        calculate_score(tree,x_test,y_test)

    for k in [5,10,15,20,25]:

        temp_RF = copy.deepcopy(RF)
        """now estimate LOF for each tree and  D I V E R S I F Y  the random forest"""
        LOFs,weights = assemble_LOFB_DRF(temp_RF,k,x_test,y_test=None)
        
        avg_tree_acc,ensemble_acc = RF_accuracies(temp_RF,x_test,y_test)
        #print(f"{avg_tree_acc} average accuracy over all trees")
        #print(f"{ensemble_acc} ensemble accuracy")

        #visualize.tree_to_png(RF.estimators_[25],"tree")
    
    
