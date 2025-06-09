################################################################################
#      VERSION 2: where scenarios grouped into 5 tasks with each OLUs that behave the same way
################################################################################

#!/usr/bin/env python
# coding: utf-8


# ─────────────────────────────────────────────────────────────────────
#                   Import necessary libraries
# ─────────────────────────────────────────────────────────────────────

import os, sys, re, time
import tensorflow as tf
import numpy as np
import keras
import cv2                                # for resizing 
from datetime import datetime
from models import CASPIANv2_MAML as new_models
from collections import defaultdict # to use a special kind of dictionary where you don t have to check if a key already exists before adding to it


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging


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

# create a dictionary where each key is a task ID from task_groups and each value is an empty list which will store filenames for the support/query set of that task 
task_support = {i: [] for i in task_groups}        # support set = mostly unprotected focus OLUs
task_query = {i: [] for i in task_groups}          # query set = mostly protected focus OLUs
# now: task_support and task_query == { 1: [], 2: [], 3: [], 4: [], 5: [] } 

used_scenarios = set () # a set is a list that only stores unique items and it's optimized for fast lookup
max_per_set = 20 # max support/query set scenarios per task 


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
    mask = np.load("./data/mask/sf_full_grid_1024.npy", allow_pickle=True).item()
    the_mask = np.zeros((1024, 1024), dtype=bool)
    idx = np.array(list(mask.values()))
    the_mask[idx[:, 0], idx[:, 1]] = True
    return the_mask.T

def get_ad_mask():
    mask = np.load("./data/mask/ad_full_grid_1024.npy", allow_pickle=True).item()
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
    print ("f_str is:", f_str)

    olu_binary = f_str.split ('_')[0]   # Extract 17-bit binary OLU string

    # Extract the SLR value (e.g. 1.5m -> 1.5)
    slr_var = 1.0 # Default value if parsing fails 
    match = re.search(r'_(\d+(\.\d+)?)m', f_str)    # Searches the filename for a pattern like _1.5m to extract the sea level rise value.

    if match:                                       # If a match is found, extracts the number (e.g., 1.5) and converts it to a float 
        slr_val = float(match.group(1)) 


    # Identify if the scenario is from AD or SF 
    if len(olu_binary) == 30: 
        region_mask = sf_mask
    elif len(olu_binary) == 17: 
        region_mask = ad_mask

    total_olus = len(olu_binary)
    
    ##### now we have the scneario's: olu_binary, slr_val, and region_mask #####
    ##### goal -> assign this scnenario to a task's support set or query set ##### 


    if f_str not in used_scenarios:
        # Loop over all tasks 
        for task_id, focus_olus in task_groups.items(): 

            # Skip to the next task if both support & query sets are full for this task 
            if len(task_support[task_id])>= max_per_set and len(task_query[task_id]) >= max_per_set: 
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
            
            # Decide which set we're assign the current scenario to 
            if protected_count < total_focus / 2:       # meaning the majority of focus OLUs are unprotected 
                target_set = task_support[task_id]      # append this scenario to the current task's support set 
                target_label = "support"
            
            elif protected_count > total_focus / 2:     # meaning protected OLUs are the majority 
                target_set = task_query [task_id]       # append this scenario to the current task's query set
                target_label = "query"
            else: 
                continue # if a scenario has protected=unprotected OLUs for the focus OLU of a task -> move to the next task  


            if len (target_set)<max_per_set: 
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
                        break # stop when you find a scenario not similar to current focus scenario & move to next task 
                
                # if in target set all the scenarios are similar to the current scenario (similar_enough = True) append to target set
                if similar_enough: 
                    target_set.append(f_str)
                    used_scenarios.add(f_str)
                    break # successfully assigned the scenario, now move to next scenario
            
    # once the scenario is assigned to a task -> rest of the code - unchanged from original clear_ds () ####
    
    x_np = x.numpy()
    y_np = y.numpy()

    # Expand dims from (1024,1024) to (1024,1024,1)
    x = tf.expand_dims(x, axis=-1)
    y = tf.expand_dims(y, axis=-1)

    return x, y, region_mask, slr_val



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


def get_all_filenames_in_dataset(ds_path): 
    """
    Load the full dataset once and return a set of all scenario filenames. 
    """
    raw_ds = tf.data.experimental.load(ds_path)
    all_filenames = set()                           # create a set of all the filenames (list with easy lookup + no duplicates)

    for f, x, y, yf in raw_ds: 
        fname = f.numpy().decode("UTF-8")
        all_filenames.add(fname)
    
    return all_filenames



################################################################################
#                    FULL SCENARIO instead of PATCH-BASED SAMPLING
################################################################################

# function that replaces sample_support_and_query_patches() [ only function changed from original code]
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
    dataset_path = "./data/AD_slr_0.5m/train/FSL_MAML_AD_train_ds_aug_split_1"

    # First collect all filenames in dataset (no repeat/shuffle here)
    all_scenarios = get_all_filenames_in_dataset(dataset_path)


    ds_train = load_tf_dataset(dataset_path) 
    # note assignment logic happens inside assign_scenario_to_task function via map_fn

    

    for f, x, y, region_mask, slr_val in ds_train:
        #break once all tasks are full 
        all_full = all(
            len(task_support[i]) >= max_per_set and len(task_query[i]) >= max_per_set 
            for i in task_groups
            )
        if all_full:
            break # get out of the for loop 
    

    # Print how many scenarios each task got and list them 
    for task_id in task_groups:
        support_list = task_support[task_id]
        query_list = task_query[task_id]

        print (f"\n Task {task_id}:")
        print (f"       -> {len(support_list)} support scenarios:")
        for fname in support_list: 
            print (f"           - {fname}")

        
        print (f"       -> {len(query_list)} query scenarios:")
        for fname in query_list: 
            print (f"           - {fname}")

    print (f"\n Total scenarios in dataset: {len(all_scenarios)}")
    print (f"Used in task sets: {len(used_scenarios)}")
    print (f"Unused: {len(all_scenarios - used_scenarios)}")



if __name__ == "__main__":
    main()
