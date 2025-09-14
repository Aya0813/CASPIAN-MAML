
"""
Optimized Version of: v2_MAML_SFAD_Training.py

This script is an updated version of the original v2 MAML training code.
The main enhancement is an optimized data loading pipeline to resolve CPU bottlenecks
during task preparation. By caching and preloading tensor batches for each task
group, this version significantly reduces overhead and improves GPU utilization
during training.

Hence the name: 'opt' for 'optimized'.
"""

# ─────────────────────────────────────────────────────────────────────
#                   Import necessary libraries
# ─────────────────────────────────────────────────────────────────────

import os, sys, re, time, pickle
import tensorflow as tf
import numpy as np
import keras
import cv2                                # for resizing 
import random

from datetime import datetime
from models import CASPIANv2_MAML as new_models
from collections import defaultdict # a special kind of dictionary where you don t have to check if a key already exists before adding to it

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# ─────────────────────────────────────────────────────────────────────
#                               MAML HYPERPARAMS
# ─────────────────────────────────────────────────────────────────────

# to change before running the code -> Hyperparameters bellow 
PATCH_SIZE = 1024           # full image
INNER_STEPS = 4           
INNER_LR = 0.0007      
OUTER_LR = 0.00003 
EPOCHS = 128                # watch convergence to avoid overfitting
STEPS_PER_EPOCH = 64        # keep in mind - full image mem intensive

# data specific
META_BATCH_SIZE = 5          # CRUTIAL in this case it is SET to 5 cause we have 5 task groups (constant)
SUPPORT_K = 2 
QUERY_K = 2
MAX_PER_SET = 20             # max support/query set scenarios per task 


# Initial test to confirm that the implementation runs successfully —> confirmed working
# PATCH_SIZE = 1024        # Full image (still needed for model input shape)
# INNER_STEPS = 1          # Just 1 inner loop update
# INNER_LR = 0.0001        # Safe learning rate
# OUTER_LR = 0.00001       # Conservative meta-learning rate
# META_BATCH_SIZE = 1      # Just 1 support-query pair per meta-update
# EPOCHS = 2               # Very short training run
# STEPS_PER_EPOCH = 2      # Just 2 updates per epoch
# SUPPORT_PATCHES and QUERY_PATCHES not used anymore




# to change before running the code -> MODEL_NAME and EXPERIMENT_NOTE
MODEL_NAME = "v2_opt_test3_with_val"
# EXPERIMENT_NOTE = (
#     "Main changes: "
#     "Testing upper-bound limits of model adaptation using full-resolution inputs, longer inner loops, "
#     "and large task pools to benchmark performance and GPU tolerance."
#     "INNER_STEPS → 8         : Balanced power of 2" 
#     "EPOCH  → 128            : Power of 2"
#     "STEPS_PER_EPOCH → 64    : Power of 2"
#     "MAX_PER_SET → 32        : Clean power of 2"
# )

SAVE_DIR = f"./models/trained_models/aya/{MODEL_NAME}"
log_file_path = f"{SAVE_DIR}/loss_log.txt"
os.makedirs(SAVE_DIR, exist_ok=True) # Creates folder if needed

OUT_DIR = f"./models/aya/AD_training/v2/{MODEL_NAME}/SF_FT_with_val"


AD_DATASET_VAL        = "./data/raw/AD_slr_0.5m/val/FSL_MAML_AD_val_ds_aug_split_1"  

# ─────────────────────────────────────────────────────────────────────
#     Define groups of OLUs → each group becomes a separate MAML task
# ─────────────────────────────────────────────────────────────────────


task_groups = {  # Type: dict[int, list[int]]
    1: [3,4,5,6],           # task 1 includes OLUs 3,4,5,6 
    2: [7,17,2,10,1],       # task 2 includes OLUs 7,17,etc. 
    3: [8,9], 
    4: [11, 12, 13, 14], 
    5: [16,15]
}



# # reassign NEW global containers for validation
task_support = {i: [] for i in task_groups}        # support set = mostly unprotected focus OLUs
task_query = {i: [] for i in task_groups}          # query set = mostly protected focus OLUs
used_scenarios = set () # a set is a list that only stores unique items and it's optimized for fast lookup


# ─────────────────────────────────────────────────────────────────────
#        Define helper function for scenario task splitting
# ─────────────────────────────────────────────────────────────────────

# Hamming distance = number of differing characters between 2 equal length strings 
def hamming_distance (str1, str2):
    return sum (b1 != b2 for b1, b2 in zip(str1, str2))  # zip pairs up the characters of 2 strings one by one (e.g -> ('1','1'), ('0','1')... )


# Given a list of focus OLUs and total OLUs, return indices not in the focus group 
def non_focus_indices (focus_olus, total_olus):
    return [i for i in range (total_olus) if (i+1) not in focus_olus]
# note: this is accounting for the fact that focus_olus is using 1-based numbering instead of python's 0-based numbering

# ─────────────────────────────────────────────────────────────────────
#        Helper functions to get binary region mask (SF or AD)
# ─────────────────────────────────────────────────────────────────────

def get_sf_mask():
    mask = np.load("./data/raw/mask/sf_full_grid_1024.npy", allow_pickle=True).item()
    the_mask = np.zeros((1024, 1024), dtype=bool)
    idx = np.array(list(mask.values()))
    the_mask[idx[:, 0], idx[:, 1]] = True
    return the_mask.T

def get_ad_mask():
    mask = np.load("./data/raw/mask/ad_full_grid_1024.npy", allow_pickle=True).item()
    the_mask = np.zeros((1024, 1024), dtype=bool)
    idx = np.array(list(mask.values()))
    the_mask[idx[:, 0], idx[:, 1]] = True
    return the_mask.T

# Load static masks once
sf_mask = get_sf_mask()
ad_mask = get_ad_mask()

# ─────────────────────────────────────────────────────────────────────
#       Main function that groups a scenario into the right task
# ─────────────────────────────────────────────────────────────────────


def assign_scenario_to_task (f,x,y,yf): 
    """
    Assigns a scenario to the support or query set of each task  
    based on the protection status of key OLUs. 
    """

    f_str = f.numpy().decode('UTF-8')   # Convert file tensor to string 
    print(f"\n Processing scenario: {f_str}")

    olu_binary = f_str.split ('_')[0]   # Extract 17-bit binary OLU string
    total_olus = len(olu_binary)

    ########        Extract the SLR value (e.g. 1.5m -> 1.5)       ########
    slr_var = 1.0 # Default value if parsing fails 
    match = re.search(r'_(\d+(\.\d+)?)m', f_str)    # Searches the filename for a pattern like _1.5m to extract the sea level rise value.
    if match:                                       # If a match is found, extracts the number (e.g., 1.5) and converts it to a float 
        slr_val = float(match.group(1)) 
    print(f" Extracted SLR value: {slr_val}")

    ########        Identify region (AD or SF)        ##########
    if total_olus == 30: 
        region_mask = sf_mask
        print(" Region identified: San Francisco (SF)")
    elif total_olus == 17: 
        region_mask = ad_mask
        print(" Region identified: Abu Dhabi (AD)")
    else: 
        print(f"Unknown OLU string length:{total_olus}, skipping scenario.")
        return None, None, None, None
    
    ##### now we have the scneario's: olu_binary, slr_val, and region_mask #####

    was_assigned = False # in order to track the status of each scenario (assigned to a support/query set or not)
    
    ##### goal -> assign this scenario to a task's support set or query set ##### 
    if f_str not in used_scenarios: 
        # Loop over all tasks 
        for task_id, focus_olus in task_groups.items(): 

            print(f" Trying to assign to task {task_id}...")

            # Skip to the next task if both support & query sets are full for this task 
            if len(task_support[task_id]) > MAX_PER_SET and len(task_query[task_id]) > MAX_PER_SET: 
                print(f" Task {task_id} full → skipping.")
                continue 

            # Count how many of the focus OLUs are protected 
            protected_count = 0 # start with zero protected OLUs (reinitiated for each task)

            # go through each OLU index we're focusing on 
            for olu in focus_olus: 
                bit = olu_binary [olu-1]                              # get the character at that position (either 1 or 0) [ keep in mind strings in python are 0-based indexes ]
                value = int (bit)                                     # convert it to a number: '0'-> 0 and '1'-> 1
                protected_count += value                              # add it to the total if it's 1 (protected)
            
            total_focus = len(focus_olus)                             # total_focus is the total number of focus OLUs per task
            non_focus_idx = non_focus_indices(focus_olus, total_olus) # Get indices of OLUs not part of this task's focus group 
            
            # Decide target set (which set we're assigning the current scenario to) 
            if protected_count < total_focus / 2:       # meaning the majority of focus OLUs are unprotected 
            #if protected_count > total_focus / 2:      # flipped just for experimentation 
                target_set = task_support[task_id]      # append this scenario to the current task's support set 
                target_label = "support"
            
            elif protected_count > total_focus / 2:     # meaning protected OLUs are the majority 
            #elif protected_count < total_focus / 2:    # flipped just for experimentation 
                target_set = task_query [task_id]       # append this scenario to the current task's query set
                target_label = "query"

            else: 
                print(f"↔ Equal protected/unprotected focus OLUs for task {task_id} → skipping.")
                continue # if a scenario has protected=unprotected OLUs for the focus OLU of a task -> move to the next task  

            print(f" Proposed for {target_label} set of task {task_id} (protected={protected_count}/{total_focus})")


            if len (target_set) < MAX_PER_SET:   # if we have space in target_set (query set or support set) 
                similar_enough = True # initialize (start optimistic)
                for existing_fname in target_set:
                    existing_binary = existing_fname.split('_')[0]
                    distance = hamming_distance(
                        ''.join(olu_binary[i] for i in non_focus_idx), 
                        ''.join(existing_binary[i] for i in non_focus_idx)
                    )

                    # keep on checking the different already existing scenarios in the target set
                    # as long as there is similarity keep going 
                    if distance >= 4:   # initially was > 2 (required hamming distance = 1 -> only 1 bit can be different)
                        similar_enough = False 
                        print(f" Dissimilar scenario found in {target_label} set (Hamming={distance}) → skipping task.")
                        break # stop when you find a scenario not similar to current focus scenario & move to next task 
                
                # if in target set all the scenarios are similar to the current scenario (similar_enough = True) append to target set
                # meaning -> (with distance >= 2) -> only one bit was changed OR non target bits were similar 
                if similar_enough:  
                    target_set.append(f_str)
                    used_scenarios.add(f_str)
                    print(f" Assigned to task {task_id} as {target_label} (Hamming matched)")
                    was_assigned = True
                    break # successfully assigned current scenario, now move to next 
            
            else:
                print(f" {target_label.capitalize()} set of task {task_id} is full.")
                # automatically tries the next task_id
        
        ## end of the for loop -> we went over all task groups 
    
    if not was_assigned: 
        print(f"WARNING: scenario {f_str} could not be assigned to any task.")

        ######## ####### ######## HARDCODED FOR NOW (because of the new found edge case) - will go back LATER CRUTIAL ######## ####### ########
        # reasoning? - cause there are a bunch of tasks that don't end up being used -> and we could use them to populate task. 3 query JUST SO it s not a bottleneck FOR NOW -> will GO BACK to it LATER  
        
        ### check if task 3 query is NOT FULL 
        #   THEN assign scenario to task 3 query  

        forced_task_id = 3 
        if len (task_query[forced_task_id]) < MAX_PER_SET: 
            print ("Hardcoded rule: Dealing with the OLU8/9 differ EDGE CASE -> force any scenario that could not be assigned to task 3 query set")
            target_label = "query"
            task_id = forced_task_id
            task_query[forced_task_id].append(f_str)
            used_scenarios.add(f_str)
            was_assigned = True
            print(f" Assigned to task {task_id} as {target_label} (hardcoded so task 3 query set does not stay empty)")
        
        else:
            print ("Tried to assigned this scenario to task 3 query set, but task 3 query set FULL. ")
        
        ######## ####### ######## ######## ####### ######## ######## ####### ######## 
    
    #### once the scenario is assigned to a task -> rest of the code - unchanged from original clear_ds () ####
    
    x_np = x.numpy()
    y_np = y.numpy()

    # Expand dims from (1024,1024) to (1024,1024,1)
    x = tf.expand_dims(x, axis=-1)
    y = tf.expand_dims(y, axis=-1)

    return x, y, region_mask, slr_val


# ─────────────────────────────────────────────────────────────────────
#           Rest of the functions unchanged from original code 
# ─────────────────────────────────────────────────────────────────────

def map_fn(f, x, y, yf):
    """
    For each dataset element = one scenario,
    we parse the big 1024x1024 arrays + region mask + slr. 
    """
    x, y, region_mask, slr_val = tf.py_function(
        assign_scenario_to_task, inp=[f, x, y, yf], Tout=(tf.float32, tf.float32, tf.bool, tf.float32)
    )
    x.set_shape([1024, 1024, 1])
    y.set_shape([1024, 1024, 1])
    return f, x, y, region_mask, slr_val

def load_tf_dataset(ds_path):
    """
    Loads a dataset from your .save format and applies map_fn.
    Each element is (filename, x, y, region_mask, slr_val).
    """
    raw_ds = tf.data.experimental.load(ds_path)
    ds = raw_ds.map(lambda f, x, y, yf: map_fn(f, x, y, yf))
    return ds 


# ──────────────────────────────────────────────────────────────────────────────────────────────
#      Functions for constructing preprocessed tensor batches from the dataset
# ──────────────────────────────────────────────────────────────────────────────────────────────

def build_dataset_lookup(ds_train):
    """
    Build a dictionary mapping filename → (x, y, region_mask, slr_val)
    """
    lookup = {}
    for f, x, y, region_mask, slr_val in ds_train:
        fname = f.numpy().decode("UTF-8")
        lookup[fname] = (x, y, region_mask, slr_val)
    return lookup

def prepare_batches (task_support, task_query, support_k, query_k):
    """ 
    For each task, divide its support/query set into non-overlapping batches.
    Each batch has 'support_k' support examples and 'query_k' query examples.

    e.g support_k = query_k = 2 (2 elements per batch for the support and query sets)
    [   #       support      #  #     query        #           
        (['fname1', 'fname2'], ['fname3', 'fname4']),  # batch 1
        (['fname7', 'fname8'], ['fname9', 'fname10']), # batch 2 ... batch K 
        ...
    ]

    Arguments: 
        task_support/task_query: set of all support/query scenarios for all tasks 
        `support_k`: The number of scenarios per batch in the support set
        `query_k`:   The number of scenarios per batch in the query set
        

    Returns:
        task_batches: dict with 
            Keys    → task IDs 
            Values  → list of tuples where each contains support batches and a query batches
        {
            task_id_1: [
                (support_batch_1, query_batch_1),
                (support_batch_2, query_batch_2),
                ...
                (support_batch_K, query_batch_K)
            ],
            task_id_2: [
                (support_batch_1, query_batch_1),
                (support_batch_2, query_batch_2),
                ...
            ],
            ...
        }
    """ 

    task_batches = {} # This will store the final batches for each task 

    # Get only the task IDs that appear in BOTH support and query dicts 
    # this avoids erros if one dict is missing a task ID (for whatever reason we don't have all tasks filled)
    common_task_ids = set(task_support.keys()) & set(task_query.keys())

    print(f"Preparing batches for {len(common_task_ids)} tasks: {sorted(common_task_ids)}")  #  Track how many tasks we're processing

    # Loop through each valid task ID (sorted for consistent order)
    for task_id in sorted(common_task_ids):

        # Get the list of support scenario filenames for this task 
        support_pool = task_support [task_id]

        # Get the list of query scenario filenames for this task 
        query_pool = task_query [task_id]

        # Calculate how many full (non overlapping) support batches can be made
        # Each support batch needs 'support_k' samples (same for query)
        max_support_batches = len(support_pool) // support_k
        max_query_batches = len(query_pool) // query_k

        # The number of valid batches is limited by whichever pool is smaller 
        K = min (max_support_batches, max_query_batches) # e.g K = 100, then for each task you prepare 100 meta-training batches.

        # if we can't make at least one batch, skip this task
        if K ==0:   # means either support/query is empty for this task
            print(f"Skipping task {task_id}: not enough support/query samples for batch size.")
            continue 

        print(f" Task {task_id}: creating {K} batches")  # Confirm task is included and how many batches it will have

        batches = [] # to hold the list of batches for this task 

        # Optional: The scenarios are shuffled once per task.
        # random.shuffle(support_pool)
        # random.shuffle(query_pool)

        # Loop to create K non overlapping batches
        for i in range (K):
            # Select support/query filenames from ith chunck 
            s_fnames = support_pool[i * support_k : (i+1) * support_k]
            q_fnames = query_pool[i * query_k : (i+1) * query_k]

            # Append this pair (support, query) to the task's batch list
            batches.append((s_fnames, q_fnames))
        
        # Store the list of batches under this task ID 
        task_batches [task_id] = batches
    
    # Return the FINAL dictionary of ALL task batches
    return task_batches 


def build_tensor_batches (lookup_dict, task_batches):
    """
    Convert filename task_batches into actual tensors
    by looking up each filename in the pre-loaded lookup_dict. 
    This creates a final dictionary: task_id -> list of full tensor batches. 

    task_batches: dictionary created using prepare_batches(...)
    """

    tensor_batches = {}             # Creates an empty dict to store final tensor batches for each task 

    # Loop over every task (e.g. Task 1, Task 2... Task 5)
    for  task_id, batch_list in task_batches.items(): # Loop through each task and its list of batches: so we can process every (support, query) batch for that task.
        tensor_batches[task_id] = []                  # Start an empty list of batches for this task 

        # Loop over every (support_filenames, query_filenames) pair in that task's batch list 
        for s_fnames, q_fnames in batch_list: 
            x_s_list, y_s_list = [], []               # Will hold support input/label tensors 
            x_q_list, y_q_list = [], []               # Will hold query input/label tensors 

            # ----------- Process support set filenames -------------
            for fname in s_fnames: 
                # Get the raw data from the lookup dict
                x, y, region_mask, slr_val = lookup_dict[fname]         # (1024, 1024, 1) tensors - x and y  

                # Add batch dimension to x and y                        # (1, 1024, 1024, 1)
                x = tf.expand_dims (x, axis = 0)
                y = tf.expand_dims (y, axis = 0) 

                # Process the label with the region mask                # (1, 1024, 1024, 4) numpy array 
                y_np = process_label_with_mask (y, region_mask)

                # Convert y_np (Numpy) into a tensorflow tensor
                y = tf.convert_to_tensor(y_np, dtype = tf.float32)

                # add this sample to the support lists 
                x_s_list.append(x)
                y_s_list.append(y)
            

            # ----------- Process support set filenames -------------
            for fname in q_fnames: 
                x, y, region_mask, slr_val = lookup_dict[fname]
                x = tf.expand_dims(x, axis=0)
                y = tf.expand_dims(y, axis=0)
                y_np = process_label_with_mask(y, region_mask)
                y = tf.convert_to_tensor(y_np, dtype=tf.float32)

                x_q_list.append(x)
                y_q_list.append(y)

            # ----------- Stack individual examples into full batch tensors -------------
            x_s_tensor = tf.concat (x_s_list, axis = 0)     # (batch_size, 1024, 1024, 1) 
            y_s_tensor = tf.concat (y_s_list, axis = 0)     # (batch_size, 1024, 1024, 4) 

            x_q_tensor = tf.concat(x_q_list, axis=0)
            y_q_tensor = tf.concat(y_q_list, axis=0)

            # Create slr_tensor of shape (query_batch_size, 1)
            slr_tensor = tf.ones([x_q_tensor.shape[0], 1], dtype = tf.float32) * slr_val # note in this code assuming slr_val is the same for both support and query 

            # Add this batch (support, query, slr) to the task's batch list 
            tensor_batches[task_id].append(
                (x_s_tensor, y_s_tensor, x_q_tensor, y_q_tensor, slr_tensor)
            )

    return tensor_batches

    # Return the full dict: 
    #   Key    -> task IDs 
    #   Values -> List of tuples where each tuple contains 5 Tensorflow tensors 

    # {
    #     task_id_1: [
    #         (x_s_tensor_1, y_s_tensor_1, x_q_tensor_1, y_q_tensor_1, slr_tensor_1),       # this is one batch 
    #         (x_s_tensor_2, y_s_tensor_2, x_q_tensor_2, y_q_tensor_2, slr_tensor_2),       # another batch ... 
    #         ...
    #         (x_s_tensor_K, y_s_tensor_K, x_q_tensor_K, y_q_tensor_K, slr_tensor_K)
    #     ],
    #     task_id_2: [
    #         (x_s_tensor_1, y_s_tensor_1, x_q_tensor_1, y_q_tensor_1, slr_tensor_1),
    #         ...
    #     ],
    #     ...
    # }


################################################################################
#                    FULL SCENARIO instead of PATCH-BASED SAMPLING
################################################################################

# function that replaces sample_support_and_query_patches() 
def process_label_with_mask(y_tensor, region_mask_tensor): 
    """
    Prepares the 4-channel label tensor for a single scenario:
    [normalized_y, mask, y_min, y_max], shape = (1, 1024, 1024, 4)

    Args:
        y_tensor (tf.Tensor):               shape (1, 1024, 1024, 1)
        region_mask_tensor (tf.Tensor):     shape (1024, 1024), dtype=bool

    Returns:
        np.ndarray:                         shape (1, 1024, 1024, 4)
    """

    y_np = y_tensor.numpy()[0, ..., 0]        # shape: (1024,1024)
    mask_np = region_mask_tensor.numpy()      # shape: (1024, 1024)     

    y_min = y_np.min()
    y_max = y_np.max()
    eps = 1e-8                      # a small constant used for numerical stability
    y_norm = (y_np - y_min) / (y_max - y_min + eps)

    # Expand y_norm from shape (1024, 1024) to (1024, 1024, 1) so it has a channel dimension
    # This makes it compatible for concatenation with other tensors later
    y_norm = np.expand_dims(y_norm, axis = -1) 

    # Convert the boolean region mask to float32 (True → 1.0, False → 0.0), 
    # then expand its shape to (1024, 1024, 1) for compatibility with y_norm
    mask = np.expand_dims(mask_np.astype (np.float32), axis = -1)

    # Create a full array with the same shape as y_norm, filled with y_min,
    # so every pixel/channel position carries the same min value (for denormalization later)
    y_min_full = np.full_like(y_norm, y_min, dtype = np.float32)

    # Same idea as above but filled with y_max instead 
    y_max_full = np.full_like (y_norm, y_max, dtype = np.float32)

    # Concatenate the 4 channels along the last axis (channel axis)
    # Resulting shape: (1024,1024,4)
    y_combined = np.concatenate([y_norm, mask, y_min_full, y_max_full], axis =-1)

    # Add the batch dimension at the front to get the final shape (1,1024,1024,4)
    y_combined = np.expand_dims (y_combined, axis = 0)

    return y_combined


################################################################################
#                           CUSTOM LOSS (with mask)
################################################################################

@tf.function
def custom_loss(y_true, y_pred):
    """
    y_true shape: (batch, patch_size, patch_size, 4)
      => channels: [0: y_norm, 1: mask, 2: y_min, 3: y_max]
    y_pred shape: (batch, patch_size, patch_size, 1)
    We replicate your original logic: 
      - un-normalize and compute huber/cosh/quantile with weighting
      - only for pixels where mask==1
    """
    # Extract channels
    y_norm  = y_true[:,:,:,0] # Normalized ground-truth depth
    mask    = y_true[:,:,:,1] # Valid region indicator
    y_min   = y_true[:,:,:,2] 
    y_max   = y_true[:,:,:,3]

    y_pred  = tf.squeeze(y_pred, axis=-1) # shape now (batch, patch_size, patch_size)

    # unnormalize -> y_real = y_norm * (y_max - y_min) + y_min
    denorm_true = y_norm*(y_max - y_min + 1e-8) + y_min
    denorm_pred = y_pred*(y_max - y_min + 1e-8) + y_min

    # mask
    valid_true = tf.boolean_mask(denorm_true, mask>0.5)
    valid_pred = tf.boolean_mask(denorm_pred, mask>0.5)

    error = tf.abs(valid_true - valid_pred)
    Delta = tf.clip_by_value(error, clip_value_min=0.3, clip_value_max=0.7)
    huber_loss = tf.keras.losses.Huber(delta=Delta)(valid_true, valid_pred)

    cosh_loss = tf.reduce_mean(tf.math.log(tf.cosh(valid_pred - valid_true)))

    weight = 3.0
    alphaH=0.3
    alphaC=0.5
    quantile=0.5

    errorQ = valid_true - valid_pred
    quantile_loss = tf.maximum((quantile - 1) * errorQ, quantile * errorQ)

    weighted_huber_loss = tf.where(valid_true > 0, huber_loss*weight, huber_loss)
    weighted_cosh_loss  = tf.where(valid_true > 0, cosh_loss*weight, cosh_loss)

    loss = alphaH*weighted_huber_loss + alphaC*weighted_cosh_loss + (1-alphaH-alphaC)*quantile_loss
    return tf.reduce_mean(loss)

################################################################################
#                         MAML TRAINER CLASS
################################################################################

class MAMLTrainer:
    def __init__(self, model, inner_lr, outer_lr, inner_steps, meta_batch_size):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.meta_batch_size = meta_batch_size
        self.outer_optimizer = tf.keras.optimizers.Adam(self.outer_lr)

    def get_trainable_vars(self):
        return self.model.trainable_variables

    @tf.function
    def fast_adapt(self, x_s, y_s, x_q, y_q, slr_val):
        """
        MAML adaptation for a single task:
          1) Save current weights
          2) Do 'inner_steps' on support set
          3) Evaluate query loss => compute grad wrt original weights
          4) restore original weights
        """
        meta_weights = self.get_trainable_vars()
        original_weights = [tf.identity(w) for w in meta_weights]

        # INNER LOOP
        for _ in range(self.inner_steps): 
            with tf.GradientTape() as tape:
                preds = self.model({"input_grid": x_s, "slr_input": slr_val}, training=True)
                s_loss = custom_loss(y_s, preds)
            grads = tape.gradient(s_loss, meta_weights)
            for w, g in zip(meta_weights, grads):
                w.assign_sub(self.inner_lr * g)

        # Query loss
        with tf.GradientTape() as outer_tape:
            preds_q = self.model({"input_grid": x_q, "slr_input": slr_val}, training=True)
            q_loss = custom_loss(y_q, preds_q)

        grads_query = outer_tape.gradient(q_loss, meta_weights)

        # restore original weights
        for w_old, w_new in zip(original_weights, meta_weights):
            w_new.assign(w_old)

        return q_loss, grads_query 

    def meta_update(self, grads_accum):
        """Apply accumulated gradients to meta-weights."""
        self.outer_optimizer.apply_gradients(zip(grads_accum, self.get_trainable_vars()))



################################################################################
#                              MAIN TRAINING LOOP
################################################################################


def main():
    start_time = time.time()  # ⏱️ Start timer

    print("TensorFlow version:", tf.__version__)
    print("Keras version:", keras.__version__)
    print("GPU available:", tf.test.is_gpu_available())

    # 1) Load your CASPIAN model 
    model = new_models.CASPIAN(
        input_shape=(PATCH_SIZE, PATCH_SIZE, 1),  # 1024*1024 - full image
        filters=72, input_centering=True,
        depth=4, cardinality=34, activation='tanh',
        bottleneck_depth=8, init="glorot_normal",
        sup_level=1, compression_factor=0.85,
        gr_level=4, slr_input=True
    )

    # Quick build: pass a dummy input
    dummy_x = tf.zeros([1, PATCH_SIZE, PATCH_SIZE, 1], dtype=tf.float32)
    dummy_slr = tf.zeros([1,1], dtype=tf.float32)
    _ = model({"input_grid": dummy_x, "slr_input": dummy_slr}, training=False)
    print(model.summary())

    # 2) Initialize MAML trainer
    maml = MAMLTrainer(
        model=model,
        inner_lr=INNER_LR,
        outer_lr=OUTER_LR,
        inner_steps=INNER_STEPS,
        meta_batch_size=META_BATCH_SIZE
    )

    log_file = open(log_file_path, "w")  


    ########################################
    # ─────────────────────────────────────────────────────────────────────
    #                             dataset
    # ─────────────────────────────────────────────────────────────────────

    dataset_path = "./data/AD_slr_0.5m/train/FSL_MAML_AD_train_ds_aug_split_1"

    ######################### training data #########################

    # note about task_support and task_query : 
    # load_tf_dataset() 
    # └── calls map_fn() for each data sample
    #     └── calls assign_scenario_to_task() 
    #         └── populates task_support and task_query lists for each task 

    # filename_train_data = f"./data/batched_data/AD/train_tensor_batches_SUPPORT_K_{SUPPORT_K}_QUERY_K_{QUERY_K}.pkl" # reason behind the naming: since MAML AD training code and the data batching depends on SUPPORT_K and QUERY_K

    # # create a dictionary where each key is a task ID from task_groups and each value is an empty list which will store filenames for the support/query set of that task 
    # task_support = {i: [] for i in task_groups}        # support set = mostly unprotected focus OLUs
    # task_query = {i: [] for i in task_groups}          # query set = mostly protected focus OLUs
    # # now: task_support and task_query == { 1: [], 2: [], 3: [], 4: [], 5: [] } 

    # used_scenarios = set () # a set is a list that only stores unique items and it's optimized for fast lookup

    # if os.path.exists(filename_train_data):          
    #     with open(filename_train_data, "rb") as f: # rb: open the file for Reading Binary data 
    #         tensor_batches = pickle.load(f)
    #     print (f"Loaded tensor_batches from {filename_train_data}") # why pickle? -> Works with any Python object, including nested dictionaries of NumPy arrays or TensorFlow tensors + Doesn't require converting to strings, JSON, or a TF format
    
    # else: # if the file does not exist then do the data batching
    #     # Loading & batching the TRAINING data & saving to disk
    #     ds_train = load_tf_dataset (dataset_path) 
    #     print("🔄 Building dataset lookup dictionary for the training data...")
    #     ds_lookup_train = build_dataset_lookup (ds_train)
    #     print(f"✅ Dataset lookup built with {len(ds_lookup_train)} entries.")     
    #     task_batches_train = prepare_batches (task_support  , task_query , support_k = SUPPORT_K, query_k = QUERY_K)
    #     tensor_batches = build_tensor_batches (ds_lookup_train, task_batches_train)  # full x/y tensors pre loaded

    #     # make the folder exists before saving 
    #     os.makedirs(os.path.dirname(filename_train_data), exist_ok = True)

    #     with open(filename_train_data, "wb") as f: 
    #         pickle.dump(tensor_batches, f)
        
    #     print (f"Saved tensor_batches to {filename_train_data}") 
    

    ######################### Validation data #########################

    filename_val_data = f"./data/processed/AD/val_tensor_batches_SUPPORT_K_{SUPPORT_K}_QUERY_K_{QUERY_K}_edge_case_solved.pkl" # reason behind the naming: since MAML AD Validation code and the data batching depends on SUPPORT_K and QUERY_K
    

    # task_support = {i: [] for i in task_groups}        # support set = mostly unprotected focus OLUs
    # task_query = {i: [] for i in task_groups}          # query set = mostly protected focus OLUs
    # # now: task_support and task_query == { 1: [], 2: [], 3: [], 4: [], 5: [] } 

    # used_scenarios = set () # a set is a list that only stores unique items and it's optimized for fast lookup


    if os.path.exists(filename_val_data):          
        with open(filename_val_data, "rb") as f: # rb: open the file for Reading Binary data 
            tensor_batches_val = pickle.load(f)
        print (f"Loaded tensor_batches from {filename_val_data}") # why pickle? -> Works with any Python object, including nested dictionaries of NumPy arrays or TensorFlow tensors + Doesn't require converting to strings, JSON, or a TF format

    else: # if the file does not exist then do the data batching

        # Loading & batching the VALIDATION data & saving to disk
        ds_val = load_tf_dataset (AD_DATASET_VAL) 
        print("🔄 Building dataset lookup dictionary for the validation data...")
        ds_lookup_val = build_dataset_lookup (ds_val)
        print(f"✅ Dataset lookup built with {len(ds_lookup_val)} entries.")     
        task_batches_val = prepare_batches (task_support , task_query , support_k = SUPPORT_K, query_k = QUERY_K)
        print (f"length of task_support is {len(task_support)} and task_query is {len(task_query)}")
        tensor_batches_val = build_tensor_batches (ds_lookup_val, task_batches_val)  # full x/y tensors pre loaded

        # make the folder exists before saving 
        os.makedirs(os.path.dirname(filename_val_data), exist_ok = True)

        with open(filename_val_data, "wb") as f: 
            pickle.dump(tensor_batches_val, f)
        
        print (f"Saved tensor_batches to {filename_val_data}") 


    # ───────────── Log Experiment Setup ───────────────────

    # Write experiment header
    log_file.write("========================================\n")

    log_file.write("============   DATA   ============\n")
    # log_file.write(f"Data path: {dataset_path}\n")
    log_file.write(f"Data path: {AD_DATASET_VAL}\n")
    # log_file.write(f"Data used for validation: {len(ds_lookup_val)}\n")


    log_file.write(f"PATCH_SIZE: {PATCH_SIZE}\n")
    log_file.write(f"META_BATCH_SIZE: {META_BATCH_SIZE}\n")

    log_file.write(f"SUPPORT_K: {SUPPORT_K}\n")
    log_file.write(f"QUERY_K: {QUERY_K}\n")
    log_file.write(f"MAX_PER_SET: {MAX_PER_SET}\n\n")



    log_file.write("============   MODEL   ============\n")
    #log_file.write(f"Base checkpoint path: {BASE_CHECKPOINT}\n\n")


    log_file.write("============   TRAINING   ============\n")
    log_file.write(f"INNER_STEPS: {INNER_STEPS}\n")
    log_file.write(f"INNER_LR: {INNER_LR}\n")
    log_file.write(f"OUTER_LR: {OUTER_LR}\n")
    log_file.write(f"EPOCHS: {EPOCHS}\n")
    log_file.write(f"STEPS_PER_EPOCH: {STEPS_PER_EPOCH}\n\n")


    # log_file.write(f"EXPERIMENT_NOTE: {EXPERIMENT_NOTE}\n\n")
    log_file.write("========================================\n\n")
    log_file.flush()

    # ──────────────────────────────────────────────────────────


    # ############ MAIN TRAINING LOOPS ################

    epoch_losses = [] # added to track epoch loss
    iteration = 0 
    batch_indices = {task_id: 0 for task_id in tensor_batches} # we need a separate index per task, since each task rotates through its own batches.
    for epoch in range(EPOCHS):
        meta_batch_losses = []

        for step in range(STEPS_PER_EPOCH):
            # accumulate grads across tasks in a meta-batch
            grads_accum = None
            task_count = 0

            for task_id in range(META_BATCH_SIZE):
                real_task_id = task_id + 1 # fix 0 based indexing 
                batch_index = batch_indices[real_task_id]

                # fetch batch 
                x_s, y_s, x_q, y_q, slr_tensor = tensor_batches[real_task_id][batch_index]
                
                # update index 
                batch_indices[real_task_id] = (batch_index + 1) % len (tensor_batches[real_task_id])

                # x -> (N, 1024, 1024, 1) and y -> (N, 1024, 1024, 4) (both x and y are tensors)

                # Fast adapt
                q_loss, grads_query = maml.fast_adapt(x_s, y_s, x_q, y_q, slr_tensor)

                # print ("post fast adapt") # for debugging
                meta_batch_losses.append(q_loss.numpy())

                if grads_accum is None:
                    grads_accum = grads_query
                else:
                    grads_accum = [ga + g for ga, g in zip(grads_accum, grads_query)]

                print(f"Epoch {epoch+1} Step {step+1} Task {task_count+1} | q_loss: {q_loss.numpy():.4f}\n", flush=True) # added for tracking training progress

                log_file.write(f"Epoch {epoch+1} Step {step+1} Task {task_count+1} | q_loss: {q_loss.numpy():.4f}\n")
                log_file.flush()  # Ensures it's written to disk immediately

                task_count += 1 # remember: the number of tasks is the META_BATCH_SIZE

            # done with the 5 tasks (META_BATCH_SIZE = 5)

            print ("\n")
            # after one meta-batch of tasks, average the grads & update
            grads_accum = [g / task_count for g in grads_accum]
            maml.meta_update(grads_accum)
            
            # update step loss (average query loss over all tasks in this meta-batch)
            step_loss = np.mean(meta_batch_losses[-task_count:]) # 1 step = going through all tasks (which is one meta batch)
            
            # for tracking: print in terminal and save to log file 
            print(f"Epoch {epoch+1} Step {step+1} | ⬅️ Meta-batch avg q_loss (step_loss): {step_loss:.4f}\n")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"[{timestamp}] Epoch {epoch+1} Step {step+1} | step_loss: {step_loss:.4f}\n")
            log_file.flush()  # Ensures it's written to disk immediately

            iteration += 1 # updated for every STEPS_PER_EPOCH 

        print ("\n")

        # backpropagate
        epoch_loss = np.mean(meta_batch_losses[-META_BATCH_SIZE*STEPS_PER_EPOCH:]) # Computes the mean query loss across all meta-batches (steps × tasks) for the epoch
        epoch_losses.append(epoch_loss) # to store epoch-wise loss for plotting 

        # done with all STEPS_PER_EPOCH steps -> done with 1 more epoch 
        
        # for tracking: print in terminal and save to log file 
        print(f"Epoch {epoch+1}/{EPOCHS}, epoch loss: {epoch_loss:.4f}")
        print ("\n")
        print ("\n")
        log_file.write(f"Epoch {epoch+1}, Epoch Loss: {epoch_loss:.4f}\n")
        log_file.flush()  # Ensures it's written to disk immediately


        # optionally save checkpoint
        # model.save_weights(os.path.join(SAVE_DIR, f"maml_epoch_{epoch+1}.h5"))

        # ##############################################################################################

        # After each epoch evaluate on the val set 
        # # ───── ───── ───── ───── Evaluation on Validation Set ───── ───── ───── ───── 
        
        val_losses = []                                                    # Create an empty list to store the validaton losses across tasks
        val_task_losses = {task_id: [] for task_id in tensor_batches_val}  # For every task in my validation batches, make a place to record its losses.
        
        # This creates a dictionary where: Each key is a task_id and each value is an empty list, which will later store the query losses for that task during validation
        # val_task_losses = {
        #     1: [],    # # losses for batches of task 1
        #     2: [],    # # losses for batches of task 2
        #     3: [],    # # losses for batches of task 3
        #     4: [],    # # losses for batches of task 4
        #     5: []     # # losses for batches of task 5 
        # }

        for task_id in tensor_batches_val:                                        # Loop over all validation tasks
            for x_s, y_s, x_q, y_q, slr_tensor in tensor_batches_val[task_id]:    # Loop over all batches (scenarios) for this task 

                # Run MAML fast adaption using support and query sets -> this gives back the query loss to this adapted model 
                val_q_loss, _ = maml.fast(x_s, y_s, x_q, y_q, slr_tensor)
                # note :maml.fast(...) returns two values: (query_loss, gradients)
                # But during validation, we only care about the loss (not updating weights), so we use '_' to ignore the gradients 

                val_loss = val_q_loss.numpy()       # Convert the query loss (a Tensor) to a Numpy number 
                val_losses.append(val_loss)         
                val_task_losses[task_id].append(val_loss)   
            
            # done with all batches of task_id and move to the next task 

            # gather the loss for each task group - CHECK TABLETTE FOR STEPS TO FOLLOW + rename the variables BETTER so they are more self explanatory 

        # done with all of the batches of all of the tasks
        # Compute the average query loss across all validation batches
        val_mean_loss = np.mean(val_losses)

        # Print and log - the evalidation loss for this epoch 
        print (f"Epoch {epoch+1}, Validation query loss: {val_mean_loss: .4f}\n")       # renanme so the logging and printing makes sense as well 
        #log_file_val.write (f"Epoch {epoch+1}, Validation query loss: {val_mean_loss: .4f}\n")
        #log_file_val.flush()    # Immediately write to disk (avoid buffering delays)

        # ───── ───── Save the best model ───── ─────
        # If the current validation loss is better (lower) than any previous epoch 
        if val_mean_loss < best_val_loss: 

            # delete old best if it exists 
            if 'best_model_path' in locals() and os.path.exists(best_model_path): 
                os.remove(best_model_path)
            
            best_val_loss = val_mean_loss   # Update the best loss seen so far 
            best_epoch = epoch + 1  
            best_model_path = os.path.join(OUT_DIR, f"best_model_epoch_{best_epoch}.h5") 
            model.save_weights(best_model_path) # Save the model weights to a file  

            # Print confirmation that the best model was saved 
            print("✓ Saved best model (val loss improved)")
            #log_file_val.write("✓ Saved best model (val loss improved)")
            #log_file_val.flush()
        
        # ───── ─────  end - best model saved ───── ───── 
        
        # # ───── ───── ───── ───── End of Validation for 1 epoch -> move to next epoch  ───── ───── ───── ───── 
        
    
    # ##############################################################################################
    
    # # final save
    # model.save_weights(os.path.join(SAVE_DIR, "final_maml_weights.h5"))
    # print("MAML Training Done!")

    # ⏱️ End timer + print summary
    end_time = time.time()
    duration = end_time - start_time

    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"⏱️ Total training time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")   
    log_file.write(f"Total training time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")
    log_file.flush()  

    #log_file.close()

if __name__ == "__main__":
    main()


