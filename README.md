# LOFBDRF
Pruning Method for ML Ensembles based on the Local Outlier Factor

## Dependencies
Python 3.9
Modules:  
Numpy  
cv2  
pandas  
PIL  
json  

## Usage

Change Directory that contains ground truth and labelmaps from C:\Users\jasper\Documents\HTCV_local to custom path.  
Run label_translator.py to transform RGB values to classifications.  
Run label_pruning --> saves final ensemble prediction as .tif file.  
Run check_results.py --> calculates score of the labelmap against ground truth (must be named Test_Labels.tif)
