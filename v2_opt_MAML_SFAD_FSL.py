
"""
Optimized Version of: v2_MAML_SFAD_FSL.py

This script is an updated version of the original v2 few-shot fine-tuning code.
The key improvement lies in optimizing the data loading process to mitigate CPU bottlenecks
during support and query task preparation. By precomputing and caching tensor batches for each
task group, this version ensures faster setup and more efficient GPU usage during fine-tuning.

In addition to fine-tuning, this version also includes a validation script for hyperparameter tuning,
allowing performance to be monitored on a validation support/query set across different configurations.

Note: The 'v2' in the filenames refers to the use of a task definition strategy that groups 
scenarios based on similarity in OLU (Operational Land Unit) protection configurations. 
This grouping creates more meaningful and coherent tasks for meta-learning.

Hence the name: 'opt' for 'optimized'.
"""

import os, sys, re, time, pickle
import tensorflow as tf
import numpy as np
import keras
import cv2                                # for resizing 
import random

from datetime import datetime
from models import CASPIANv2_MAML as new_models
from collections import defaultdict # a special kind of dictionary where you don t have to check if a key already exists before adding to it


# ───── Locations ────────────────────────────────────────────────────────────
# from original code 
# BASE_CHECKPOINT = "./models/trained_models/MAML_CASPIAN_patch/final_maml_weights.h5"
# SF_DATASET      = "./data/FSLFSL_MAML_SF_dataset"  # <‑‑ your file
# OUT_DIR         = "./models/trained_models/MAML_CASPIAN_patch_FT_SF"
# os.makedirs(OUT_DIR, exist_ok=True)

### 
# Part 1 to change before running the code

MODEL_NAME = "v2_opt_test1"
BASE_CHECKPOINT = f"./models/trained_models/aya/{MODEL_NAME}/final_maml_weights.h5"         

OUT_DIR = f"./models/trained_models/aya/{MODEL_NAME}/SF_FT_with_val"
log_dir = os.path.join(OUT_DIR, "logs")
log_file_path = os.path.join(log_dir, "train_log.txt")      # to log all training parameters
log_file_path_val = os.path.join(log_dir, "val_log.txt")    # to log all validation parameters

# Ensure the logs directory exists
os.makedirs(log_dir, exist_ok=True)

# ───────────────────────────── Data ────────────────────────────────────────
# Initial mistake: unintentionally used the full San Francisco dataset 
# SF_DATASET    = "./data/SF_slr_1m/FSL_MAML_SF_dataset"   

# now only using a section for training and the rest of the data val & test 
SF_DATASET_TRAIN      = "./data/SF_slr_1m/FSL_MAML_SF_train_ds_aug_split_1"  
SF_DATASET_VAL        = "./data/SF_slr_1m/FSL_MAML_SF_val_ds_aug_split_1"  


# ───── Hyper‑params for **fine‑tuning**  (much smaller / shorter) ───────────
# Part 2 to change before running the code
# same as the hyperparameters for training with the exception of FINETUNE_EPOCHS
PATCH_SIZE       = 1024        # no change
INNER_STEPS      = 5           # good balance
INNER_LR         = 0.00002     # safe low LR
OUTER_LR         = 0.000002    # stable meta updates

FINETUNE_EPOCHS  = 20          
STEPS_PER_EPOCH  = 10          


# data specific
META_BATCH_SIZE  = 5           # keep for 5 task groups
SUPPORT_K = 1
QUERY_K = 1
MAX_PER_SET = 5                # max support/query set scenarios per task 


# Part 3 (final) to change before running the code
EXPERIMENT_NOTE = ("Now validation setup working (first test was using v2_opt_test2), in this run specifically I changed the OLU groupings according to the found literature.")


# ─────────────────────────────────────────────────────────────────────
#     Define groups of OLUs → each group becomes a separate MAML task
# ─────────────────────────────────────────────────────────────────────

# initial arbitrary 
# task_groups = {  # Type: dict[int, list[int]]
#     1: [1, 2, 3, 4, 5, 6],            # Task 1: OLUs 1–6
#     2: [7, 8, 9, 10, 11, 12],         # Task 2: OLUs 7–12
#     3: [13, 14, 15, 16, 17, 18],      # Task 3: OLUs 13–18
#     4: [19, 20, 21, 22, 23, 24],      # Task 4: OLUs 19–24
#     5: [25, 26, 27, 28, 29, 30]       # Task 5: OLUs 25–30
# }

# new OLU groupings according to found literature (see final report)
task_groups = {  # Type: dict[int, list[int]]
    1: [1, 2, 3, 4, 5, 6],                  # Task 1: OLUs 1–6
    2: [7, 8, 9, 10, 11, 12],               # Task 2: OLUs 7–12
    3: [13, 14, 15, 16, 17],                # Task 3: OLUs 13–18
    4: [18, 19, 20, 21, 22],                # Task 4: OLUs 19–24
    5: [23, 24, 25, 26, 27, 28, 29, 30]     # Task 5: OLUs 25–30
}

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


# ───── Helper to load the TF‑Dataset exactly as before ──────────────────────
def get_sf_mask():
    d = np.load("./data/mask/sf_full_grid_1024.npy", allow_pickle=True).item()
    m  = np.zeros((1024,1024), bool)
    idx= np.array(list(d.values()));  m[idx[:,0],idx[:,1]] = True
    return m.T
MASK_SF = get_sf_mask()

# ─────────────────────────────────────────────────────────────────────
#       Main function that groups a scenario into the right task 
# ─────────────────────────────────────────────────────────────────────

def assign_scenario_to_task (f,x,y,yf): ## INSTEAD OF clear_ds()
    """
    Assigns a scenario to the support or query set of each task  
    based on the protection status of key OLUs. 
    """

    f_str = f.numpy().decode('UTF-8')   # Convert file tensor to string 
    print(f"\n Processing scenario: {f_str}")

    olu_binary = f_str.split ('_')[0]   # Extract 17-bit binary OLU string
    total_olus = len(olu_binary)

    ######## Extract the SLR value (e.g. 1.5m -> 1.5) ########
    slr_var = 1.0 # Default value if parsing fails 
    match = re.search(r'_(\d+(\.\d+)?)m', f_str)    # Searches the filename for a pattern like _1.5m to extract the sea level rise value.
    if match:                                       # If a match is found, extracts the number (e.g., 1.5) and converts it to a float 
        slr_val = float(match.group(1)) 
    print(f" Extracted SLR value: {slr_val}")

    ########          Identify region (AD or SF)     ##########
    if total_olus == 30: 
        region_mask = MASK_SF
        print(" Region identified: San Francisco (SF)")
    elif total_olus == 17: 
        region_mask = ad_mask
        print(" Region identified: Abu Dhabi (AD)")
    else: 
        print (f"Unknown OLU string length:{total_olus}, skipping scenario.")
        return None, None, None, None
    
    ##### now we have the scneario's: olu_binary, slr_val, and region_mask #####

    was_assigned = False # in order to track the status of each scenario (assigned to a support/query set or not)
    
    ##### goal -> assign this scenario to a task's support set or query set ##### 
    if f_str not in used_scenarios:
        # Loop over all tasks 
        for task_id, focus_olus in task_groups.items(): 

            print(f" Trying to assign to task {task_id}...")

            # Skip to the next task if both support & query sets are full for this task 
            if len(task_support[task_id])> MAX_PER_SET and len(task_query[task_id]) > MAX_PER_SET: 
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
                target_set = task_support[task_id]      # append this scenario to the current task's support set 
                target_label = "support"
            
            elif protected_count > total_focus / 2:     # meaning protected OLUs are the majority 
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
                    if distance >= 2:
                        similar_enough = False 
                        print(f" Dissimilar scenario found in {target_label} set (Hamming={distance}) → skipping task.")
                        break # stop when you find a scenario not similar to current focus scenario & move to next task 
                
                # if in target set all the scenarios are similar to the current scenario (similar_enough = True) append to target set
                if similar_enough: 
                    target_set.append(f_str)
                    used_scenarios.add(f_str)
                    print(f" Assigned to task {task_id} as {target_label} (Hamming matched)")
                    was_assigned = True
                    break # successfully assigned current scenario, now move to next 
            
            else:
                print(f" {target_label.capitalize()} set of task {task_id} is full.")
                # automatically tries the next task_id
    
    if not was_assigned: 
        print(f"WARNING: scenario {f_str} could not be assigned to any task.")
    
    ## once the scenario is assigned to a task -> rest of the code - unchanged from original clear_ds () ####
    
    x_np = x.numpy()
    y_np = y.numpy()

    # Expand dims from (1024,1024) to (1024,1024,1)
    x = tf.expand_dims(x, axis=-1)
    y = tf.expand_dims(y, axis=-1)

    return x, y, region_mask, slr_val

def map_fn(f,x,y,yf):
    x,y,mask,slr = tf.py_function(assign_scenario_to_task,[f,x,y,yf],
                                  [tf.float32,tf.float32,tf.bool,tf.float32])
    x.set_shape([1024,1024,1]); y.set_shape([1024,1024,1])
    return f,x,y,mask,slr

def load_tf_dataset(path):
    return tf.data.Dataset.load(path).map(map_fn)


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

    e.g support_k = query_k = 2 
    [
        (['fname1', 'fname2'], ['fname3', 'fname4']),  # batch 1
        (['fname7', 'fname8'], ['fname9', 'fname10']), # batch 2 ... batch K 
        ...
    ]

    Arguments: 
        task_support/task_query     : set of all support/query scenarios for all tasks 
        support_k/query_k           : how many support scenarios you use to adapt the model for a task.
                                    i.e. The number of scenarios per batch in the support set (same for query)

    Returns:
        task_batches: dict with task_id → list of (support_fnames, query_fnames)
    """ 

    task_batches = {} # This will store the final batches for each task 

    # Get only the task IDs that appear in BOTH support and query dicts 
    # this avoids erros if one dict is missing a task ID (for whatever reason we don t have all tasks filled)
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

        # if we can t make at least one batch, skip this task
        if K ==0: 
            print(f"Skipping task {task_id}: not enough support/query samples for batch size.")
            continue 

        print(f" Task {task_id}: creating {K} batches")  # Confirm task is included and how many batches it will have


        batches = [] # to hold the list of batches for this task 

        # The scenarios are shuffled once per task.
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



################################################################################
#                    NEW ADDED FUNCTION TO OPTIMIZE THE DATA LOADING 
################################################################################

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

    # Return the full dict: task_id -> list of (x_s, y_s, x_q, y_q, slr_tensor)
    return tensor_batches

# task_batches = {
#     1: [ (support_fnames_1, query_fnames_1), (support_fnames_2, query_fnames_2), ... ],
#     2: [ (support_fnames_1, query_fnames_1), ... ],
#     ...
# }

# Each item in a batch list is a tuple: (support_fnames, query_fnames) — where each of those is a list of filenames


# ───── process_label_with_mask instead of sample_support_query (unchanged from v1 MAML training code)  ───────────────────
def process_label_with_mask(y_tensor, region_mask_tensor): 
    """
    Prepares the 4-channel label tensor for a single scenario:
    [normalized_y, mask, y_min, y_max], shape = (1, 1024, 1024, 4)

    Args:
        y_tensor (tf.Tensor): shape (1, 1024, 1024, 1)
        region_mask_tensor (tf.Tensor): shape (1024, 1024), dtype=bool

    Returns:
        np.ndarray: shape (1, 1024, 1024, 4)
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


# ───── Custom loss (copy‑paste from your training script) ───────────────────
@tf.function
def custom_loss(y_t,y_p):
    y_n,mask,y_min,y_max = y_t[:,:,:,0],y_t[:,:,:,1],y_t[:,:,:,2],y_t[:,:,:,3]
    y_p = tf.squeeze(y_p,-1); true = y_n*(y_max-y_min+1e-8)+y_min
    pred= y_p*(y_max-y_min+1e-8)+y_min
    vt,vp = tf.boolean_mask(true,mask>0.5), tf.boolean_mask(pred,mask>0.5)
    delta = tf.clip_by_value(tf.abs(vt-vp),0.3,0.7)
    huber = tf.keras.losses.Huber(delta=delta)(vt,vp)
    cosh  = tf.reduce_mean(tf.math.log(tf.cosh(vp-vt)))
    w     = 3.0; alphaH,alphaC,quant = 0.3,0.5,0.5
    qloss = tf.maximum((quant-1)*(vt-vp), quant*(vt-vp))
    loss  = alphaH*huber*w + alphaC*cosh*w + (1-alphaH-alphaC)*qloss
    return tf.reduce_mean(loss)

# ───── MAML trainer (same as before) ────────────────────────────────────────
class MAMLTrainer:
    def __init__(s,model,inner_lr,outer_lr,inner_steps,batch):
        s.m=model; s.ilr=inner_lr; s.olr=outer_lr
        s.inner_steps=inner_steps; s.batch=batch
        s.opt=tf.keras.optimizers.Adam(outer_lr)
    def fast(s,xs,ys,xq,yq,slr):
        vars    = s.m.trainable_variables
        backup  = [tf.identity(v) for v in vars]
        for _ in range(s.inner_steps):
            with tf.GradientTape() as t:
                p_s = s.m({"input_grid":xs,"slr_input":slr},training=True)
                l_s = custom_loss(ys,p_s)
            g = t.gradient(l_s, vars)
            for v, gv in zip(vars, g):
                if gv is not None:            # skip variables that received no grad
                    v.assign_sub(s.ilr * gv)
                        
        with tf.GradientTape() as tq:
            p_q = s.m({"input_grid":xq,"slr_input":slr},training=True)
            l_q = custom_loss(yq,p_q)
        g_q = tq.gradient(l_q, vars)
        for v,b in zip(vars,backup): v.assign(b)
        return l_q, g_q
    def meta_update(s,g_acc):
        s.opt.apply_gradients(zip(g_acc, s.m.trainable_variables))

# ───── Build CASPIAN & load **pre‑trained** weights ─────────────────────────

start_time = time.time()  # ⏱️ Start timer

model = new_models.CASPIAN(
    input_shape=(PATCH_SIZE,PATCH_SIZE,1),
    filters=72,input_centering=True,depth=4,cardinality=34,activation='tanh',
    bottleneck_depth=8,init="glorot_normal",sup_level=1,
    compression_factor=0.85,gr_level=4,slr_input=True)
_ = model({"input_grid":tf.zeros([1,PATCH_SIZE,PATCH_SIZE,1]),
           "slr_input":tf.zeros([1,1])})
model.load_weights(BASE_CHECKPOINT)
print("✓  base MAML weights loaded")

maml = MAMLTrainer(model, INNER_LR, OUTER_LR, INNER_STEPS, META_BATCH_SIZE)

# ─────────────────────────────────────────────────────────────────────
#                               3) dataset
# ─────────────────────────────────────────────────────────────────────

######################### training data #########################

filename_train_data = f"./data/batched_data/SF/train_tensor_batches_SUPPORT_K_{SUPPORT_K}_QUERY_K_{QUERY_K}.pkl" # reason behind the naming: since MAML AD training code and the data batching depends on SUPPORT_K and QUERY_K

# create a dictionary where each key is a task ID from task_groups and each value is an empty list which will store filenames for the support/query set of that task 
task_support = {i: [] for i in task_groups}        # support set = mostly unprotected focus OLUs
task_query = {i: [] for i in task_groups}          # query set = mostly protected focus OLUs
# now: task_support and task_query == { 1: [], 2: [], 3: [], 4: [], 5: [] } 
used_scenarios = set () # a set is a list that only stores unique items and it's optimized for fast lookup

if os.path.exists(filename_train_data):          
    with open(filename_train_data, "rb") as f: # rb: open the file for Reading Binary data 
        tensor_batches_train = pickle.load(f)
    print (f"Loaded tensor_batches from {filename_train_data}") # why pickle? -> Works with any Python object, including nested dictionaries of NumPy arrays or TensorFlow tensors + Doesn't require converting to strings, JSON, or a TF format


else: # if the file does not exist then do the data batching

    # Loading & batching the TRAINING data & savint to disk
    ds_train = load_tf_dataset (SF_DATASET_TRAIN) 
    print("🔄 Building dataset lookup dictionary for the training data...")
    ds_lookup_train = build_dataset_lookup (ds_train)
    print(f"✅ Dataset lookup built with {len(ds_lookup_train)} entries.")     
    task_batches_train = prepare_batches (task_support  , task_query , support_k = SUPPORT_K, query_k = QUERY_K)
    tensor_batches_train = build_tensor_batches (ds_lookup_train, task_batches_train)  # full x/y tensors pre loaded

    # make the folder exists before saving 
    os.makedirs(os.path.dirname(filename_train_data), exist_ok = True)

    with open(filename_train_data, "wb") as f: 
        pickle.dump(tensor_batches_train, f)
    
    print (f"Saved tensor_batches to {filename_train_data}") 



######################### Validation data #########################

filename_val_data = f"./data/batched_data/SF/val_tensor_batches_SUPPORT_K_{SUPPORT_K}_QUERY_K_{QUERY_K}.pkl" # reason behind the naming: since MAML AD Validation code and the data batching depends on SUPPORT_K and QUERY_K


# reassign NEW global containers for validation
task_support = {i: [] for i in task_groups}        # support set = mostly unprotected focus OLUs
task_query = {i: [] for i in task_groups}          # query set = mostly protected focus OLUs
used_scenarios = set () # a set is a list that only stores unique items and it's optimized for fast lookup



if os.path.exists(filename_val_data):          
    with open(filename_val_data, "rb") as f: # rb: open the file for Reading Binary data 
        tensor_batches_val = pickle.load(f)
    print (f"Loaded tensor_batches from {filename_val_data}") # why pickle? -> Works with any Python object, including nested dictionaries of NumPy arrays or TensorFlow tensors + Doesn't require converting to strings, JSON, or a TF format


else: # if the file does not exist then do the data batching

    # Loading & batching the VALIDATION data & saving to disk
    ds_val = load_tf_dataset (SF_DATASET_VAL) 
    print("🔄 Building dataset lookup dictionary for the validation data...")
    ds_lookup_val = build_dataset_lookup (ds_val)
    print(f"✅ Dataset lookup built with {len(ds_lookup_val)} entries.")     
    task_batches_val = prepare_batches (task_support , task_query , support_k = SUPPORT_K, query_k = QUERY_K)
    tensor_batches_val = build_tensor_batches (ds_lookup_val, task_batches_val)  # full x/y tensors pre loaded

    # make the folder exists before saving 
    os.makedirs(os.path.dirname(filename_val_data), exist_ok = True)

    with open(filename_val_data, "wb") as f: 
        pickle.dump(tensor_batches_val, f)
    
    print (f"Saved tensor_batches to {filename_val_data}") 


# # ───────────── Log Experiment Setup ────────────────────────────────────────

log_file = open(log_file_path, "w")  
log_file_val = open(log_file_path_val, "w")  

# Write experiment header
log_file.write("========================================\n")

log_file.write(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

log_file.write("============   DATA   ============\n")

log_file.write(f"Training Data path: {SF_DATASET_TRAIN}\n")
# log_file.write(f"Data used for training: {len(ds_lookup_train)}\n") # LATER - log in the number of elements used 

log_file.write(f"Validation Data path: {SF_DATASET_VAL}\n")
# log_file.write(f"Data used for validation: {len(ds_lookup_val)}\n") 


log_file.write(f"PATCH_SIZE: {PATCH_SIZE}\n")
log_file.write(f"META_BATCH_SIZE: {META_BATCH_SIZE}\n")

log_file.write(f"SUPPORT_K: {SUPPORT_K}\n")
log_file.write(f"QUERY_K: {QUERY_K}\n")
log_file.write(f"MAX_PER_SET: {MAX_PER_SET}\n")


log_file.write("============   MODEL   ============\n")
log_file.write(f"Base checkpoint path: {BASE_CHECKPOINT}\n")


log_file.write("============   TRAINING   ============\n")
log_file.write(f"INNER_STEPS: {INNER_STEPS}\n")
log_file.write(f"INNER_LR: {INNER_LR}\n")
log_file.write(f"OUTER_LR: {OUTER_LR}\n")
log_file.write(f"FINETUNE_EPOCHS: {FINETUNE_EPOCHS}\n")
log_file.write(f"STEPS_PER_EPOCH: {STEPS_PER_EPOCH}\n")


log_file.write(f"EXPERIMENT_NOTE: {EXPERIMENT_NOTE}\n")
log_file.write("========================================\n")
log_file.flush()

# ─────────────── Fine‑tuning loop (few epochs) ────────────────────────────────────────


best_val_loss = float("inf")        # initialized as a very big number (to ensure that we always keep improving)
batch_indices = {task_id: 0 for task_id in tensor_batches_train} # we need a separate index per task, since each task rotates through its own batches.

for epoch in range(FINETUNE_EPOCHS):
    losses = []
    for step in range(STEPS_PER_EPOCH):
        acc_g, n = None, 0
        for task_id in range(META_BATCH_SIZE):

            real_task_id = task_id + 1 # fix 0 based indexing 
            batch_index = batch_indices[real_task_id]

            # fetch batch 
            x_s, y_s, x_q, y_q, slr_tensor = tensor_batches_train[real_task_id][batch_index]
            
            # update index 
            batch_indices[real_task_id] = (batch_index + 1) % len (tensor_batches_train[real_task_id])

            # x -> (N, 1024, 1024, 1) and y -> (N, 1024, 1024, 4) (both x and y are tensors)

            qloss,g = maml.fast(x_s,y_s,x_q,y_q,slr_tensor)

            print(f"Epoch {epoch+1}/{FINETUNE_EPOCHS}, Step {step+1}/{STEPS_PER_EPOCH}, Task {n+1}/{META_BATCH_SIZE}, query-loss: {qloss:.4f}\n")
            log_file.write(f"Epoch {epoch+1}/{FINETUNE_EPOCHS}, Step {step+1}/{STEPS_PER_EPOCH}, Task {n+1}/{META_BATCH_SIZE}, query-loss: {qloss:.4f}\n")
            log_file.flush()  # Ensures it's written to disk immediately

            losses.append(qloss.numpy())

            if acc_g is None: 
                acc_g=g
            else:             
                acc_g=[a+b for a,b in zip(acc_g,g)]
            
            n+=1  # in the main training code this is task_count

        acc_g=[g/n for g in acc_g]
        maml.meta_update(acc_g)

    print(f"\nEpoch {epoch+1}/{FINETUNE_EPOCHS}, mean query-loss: {np.mean(losses):.4f}\n")
    log_file.write(f"\nEpoch {epoch+1}/{FINETUNE_EPOCHS}, mean query-loss: {np.mean(losses):.4f}\n\n")
    log_file.flush()  # Ensures it's written to disk immediately


    # ───── ───── ───── ───── Evaluation on Validation Set ───── ───── ───── ───── 
    val_losses = []                                                    # Create an empty list to store the validaton losses for each task 

    for task_id in tensor_batches_val:                                        # Loop over all validation tasks
        for x_s, y_s, x_q, y_q, slr_tensor in tensor_batches_val[task_id]:    # Loop over all batches (scenarios) for this task 

            # Run MAML fast adaption using support and query sets -> this gives back the query loss to this adapted model 
            val_q_loss, _ = maml.fast(x_s, y_s, x_q, y_q, slr_tensor)
            # note :maml.fast(...) returns two values: (query_loss, gradients)
            # But during validation, we only care about the loss (not updating weights), so we use '_' to ignore the gradients 

            # Convert the query loss (a Tensor) to a Numpy number and store it 
            val_losses.append(val_q_loss.numpy())
        
    # Compute the average query loss across all validation batches
    val_mean_loss = np.mean(val_losses)

    # Print the evalidation loss for this epoch 
    print (f"Epoch {epoch+1}, Validation query loss: {val_mean_loss: .4f}\n")

    # Log the validation loss to a fine for later analysis 
    log_file_val.write (f"Epoch {epoch+1}, Validation query loss: {val_mean_loss: .4f}\n")
    log_file_val.flush()    # Immediately write to disk (avoid buffering delays)

    # ───── Save the best model ─────
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
            

# ───── Save fine‑tuned weights ──────────────────────────────────────────────
FT_WEIGHTS = os.path.join(OUT_DIR, "final_maml_ft_sf.h5")
model.save_weights(FT_WEIGHTS)
print("✓  fine‑tuned weights written to", FT_WEIGHTS)
log_file.write(f"fine‑tuned weights written to {FT_WEIGHTS}\n")
log_file.flush()  # Ensures it's written to disk immediately


# ⏱️ End timer + print summary
end_time = time.time()
duration = end_time - start_time

hours, rem = divmod(duration, 3600)
minutes, seconds = divmod(rem, 60)

print(f"⏱️ Total training time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")   
log_file.write(f"Total training time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")
log_file.flush()  

log_file.close()
log_file_val.close()
