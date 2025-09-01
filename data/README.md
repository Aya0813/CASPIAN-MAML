# Data Folder Structure

This folder stores all datasets used in CASPIAN-MAML experiments.

## Layout

```
data/
├── raw/        # Original .npy files (AD, SF full grids)
├── processed/  # TensorFlow datasets saved 
└── README.md
```


## Contents
- **raw/**  
  - **mask/**  
    - `ad_full_grid_1024.npy` → grid mask for Abu Dhabi  
    - `sf_full_grid_1024.npy` → grid mask for San Francisco  

  - **AD_slr_0.5m/**      
    - `archive/`  → contains the original dataset (before and after feeding it to the data construction script)
    - `test/`            
    - `val/`
    - `train/`   
  
  - **SF_slr_1m/**      
    - `archive/`  → contains the original dataset (before and after feeding it to the data construction script)
    - `test/`            
    - `val/`
    - `train/`   

- **processed/**  
  - **AD/**      
    - `.out files`  → log file
    - `.pkl files`  → saves the actual batched data 
  - **SF/**      
    - `.out files`  
    - `.pkl files` 

## Understanding how the processed dataset came to be 
- Explanation of the process for batching (the hyperparameters, the hamming distance...)
  - ... 
- Explanation of how is it used in training 
  - ... 


## Note 

- A PKL file is a file saved in the Python pickle format, which contains serialized Python objects. These files are typically used to store machine learning models, data pre-processing objects, or any Python objects that need to be saved for later use.