################################################################################
# VERSION 1 (v1): Each task consists of a single support scenario and a single query scenario
################################################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import tensorflow as tf
import numpy as np
import keras
import re
from datetime import datetime
import time

# added for debugging 
import sys

# added for resizing 
import cv2

# Import your CASPIAN model
from models import CASPIANv2_MAML as new_models

################################################################################
#                              MAML HYPERPARAMS
################################################################################

# to change before running the code
PATCH_SIZE = 1024           # full image
INNER_STEPS = 10            # increase to 10-20 to adapt meaningfully from a single full image
INNER_LR = 0.00002      
OUTER_LR = 0.000001 
META_BATCH_SIZE = 5        # 2-4 - cause full image mem intensive
EPOCHS = 50                # 50 - 100 - watch convergence to avoid overfitting
STEPS_PER_EPOCH = 25       # 20-40 - cause full image mem intensive

# Initial test to confirm that the implementation runs successfully —> confirmed working
# PATCH_SIZE = 1024        # Full image (still needed for model input shape)
# INNER_STEPS = 1          # Just 1 inner loop update
# INNER_LR = 0.0001        # Safe learning rate
# OUTER_LR = 0.00001       # Conservative meta-learning rate
# META_BATCH_SIZE = 1      # Just 1 support-query pair per meta-update
# EPOCHS = 2               # Very short training run
# STEPS_PER_EPOCH = 2      # Just 2 updates per epoch

# SUPPORT_PATCHES and QUERY_PATCHES not used anymore

# to change before running the code
MODEL_NAME = "v1_test5"
SAVE_DIR = f"./models/trained_models/aya/{MODEL_NAME}"

log_file_path = f"{SAVE_DIR}/loss_log.txt"

os.makedirs(SAVE_DIR, exist_ok=True) # Creates folder if needed


################################################################################
#                     LOAD & PREPARE EACH SCENARIO
################################################################################


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

sf_mask = get_sf_mask()
ad_mask = get_ad_mask()

def clear_ds(f, x, y, yf):
    """
    Similar to your original code:
      - We parse the SLR from filename
      - Determine if it's SF or AD by checking filename length
      - Return the full (x,y,mask,slr)
    """
    f_str = f.numpy().decode('UTF-8')               # Converts the filename f into a regular Python string.
    # print ("f is", f)                              # prints e.g -> tf.Tensor(b'100000....1_0.5m.input.npy_rnd30_v4', shape =(), dtype = string)
    print ("f_str is", f_str)                      # prints e.g -> 100000....1_0.5m.input.npy_rnd30_v4
    
    slr_val = 1.0                                   # Initializes slr_val to a default value of 1.0 in case parsing fails.
    match = re.search(r'_(\d+(\.\d+)?)m', f_str)    # Searches the filename for a pattern like _1.5m to extract the sea level rise value.
    if match:                                       # If a match is found, extracts the number (e.g., 1.5) and converts it to a float 
        slr_val = float(match.group(1))

    # Identify region
    if len(f_str.split('_')[0]) == 30: # if 30 OLUs -> SF (if not AD)
        region_mask = sf_mask
    else:
        region_mask = ad_mask

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
        clear_ds, inp=[f, x, y, yf], Tout=(tf.float32, tf.float32, tf.bool, tf.float32)
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


################################################################################
#                    FULL SCENARIO instead of PATCH-BASED SAMPLING
################################################################################

# function that replaces sample_support_and_query_patches() [ only function changed from original code ]
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

    # 3) Load your dataset. 
    data_path = "./data/AD_slr_0.5m/train/FSL_MAML_AD_train_ds_aug_split_1" # to change in case needed
    ds_train = load_tf_dataset(data_path)

    # instead of pulling one scenario at a time we're pulling 2 scenarios at a time
    ds_train = ds_train.shuffle(200, reshuffle_each_iteration=True).batch(2).repeat()
    ds_iter = iter(ds_train) # now next(ds_iter) will return a batch of 2 scenarios instead of 1 


    # Write experiment header
    log_file.write("========================================\n")
    log_file.write(f"Data used: {data_path}\n")
    log_file.write(f"Model Name: {MODEL_NAME}\n")
    log_file.write(f"PATCH_SIZE: {PATCH_SIZE}\n")
    log_file.write(f"INNER_STEPS: {INNER_STEPS}\n")
    log_file.write(f"INNER_LR: {INNER_LR}\n")
    log_file.write(f"OUTER_LR: {OUTER_LR}\n")
    log_file.write(f"META_BATCH_SIZE: {META_BATCH_SIZE}\n")
    log_file.write(f"EPOCHS: {EPOCHS}\n")
    log_file.write(f"STEPS_PER_EPOCH: {STEPS_PER_EPOCH}\n")
    log_file.write("========================================\n\n")
    log_file.flush()
' '
    epoch_losses = [] # added to track epoch loss
    iteration = 0 
    for epoch in range(EPOCHS):
        meta_batch_losses = []

        for step in range(STEPS_PER_EPOCH):
            # accumulate grads across tasks in a meta-batch
            grads_accum = None
            task_count = 0

            for _ in range(META_BATCH_SIZE):

                # 2 scenarios instead of 1 (1 for support set and 1 for query set)
                f_str_batch, x_big_batch, y_big_batch, region_mask_batch, slr_val_batch = next (ds_iter)
                # f_str_batch       -> (2,) ← 1 string per scenario
                # x_big_batch       -> (2, 1024, 1024, 1) and y_big_batch -> (2, 1024, 1024, 1)
                # slr_val_batch     -> (2,) ← 1 float per scenario
                # region_mask_batch -> (2, 1024, 1024) ← batch of bool masks ✅
            

                # get the support set instance 
                x_s = x_big_batch [0:1, ...]                        # shape: (1,1024,1024,1)
                y_s = y_big_batch [0:1, ...]                        # shape: (1,1024,1024,1)
                region_mask_s = region_mask_batch [0]               # shape: (1024, 1024) bool
                slr_val_s = slr_val_batch[0]                        # shape: 1 float value


                # get the query set instance
                x_q = x_big_batch [1:2, ...]                         # shape: (1,1024,1024,1)
                y_q = y_big_batch [1:2, ...]                         # shape: (1,1024,1024,1)
                region_mask_q = region_mask_batch [1]                # shape: (1024, 1024) bool 
                slr_val_q = slr_val_batch[1]                         # shape: 1 float value

            
                y_s_final_np = process_label_with_mask(y_s, region_mask_s)
                y_q_final_np = process_label_with_mask(y_q, region_mask_q)
                # y final -> shape: (1, 1024, 1024, 4)

                # convert y from numpy to tensor (x is already a tensor) 
                y_s = tf.convert_to_tensor(y_s_final_np, dtype=tf.float32)
                y_q = tf.convert_to_tensor(y_q_final_np, dtype=tf.float32)

                # x -> (1, 1024, 1024, 1) and y -> (1, 1024, 1024, 4) (both x and y are tensors now)

                # SLR 
                slr_tensor = tf.reshape(slr_val_s, [1, 1])  # shape: (1,1)
                #slr_tensor = tf.reshape(slr_val_q, [1, 1])  # shape: (1,1)
                
                # IMPORTANT NOTEE: this code assumes SAME SLR - however - in case this code is used with scenarios with different SLR values then the use of the split_TF_dataset_by_SLR.py is NECESSARY cause there cannot be a mismatch in training
                
                # Fast adapt
                q_loss, grads_query = maml.fast_adapt(x_s, y_s, x_q, y_q, slr_tensor)
                meta_batch_losses.append(q_loss.numpy())

                if grads_accum is None:
                    grads_accum = grads_query
                else:
                    grads_accum = [ga + g for ga, g in zip(grads_accum, grads_query)]

                print(f"Epoch {epoch+1} Step {step+1} Task {task_count+1} | q_loss: {q_loss.numpy():.4f}\n") # added for tracking training progress

                log_file.write(f"Epoch {epoch+1} Step {step+1} Task {task_count+1} | q_loss: {q_loss.numpy():.4f}\n")
                log_file.flush()  # Ensures it's written to disk immediately

                task_count += 1 # remember: the number of tasks is the META_BATCH_SIZE

            # after one meta-batch of tasks, average the grads & update
            grads_accum = [g / task_count for g in grads_accum]
            maml.meta_update(grads_accum)
            
            # update step loss
            step_loss = np.mean(meta_batch_losses[-task_count:]) # 1 step = going through all tasks (which is one meta batch)
            
            # for tracking: print in terminal and save to log file 
            print(f"Epoch {epoch+1} Step {step+1} | ⬅️ Meta-batch avg q_loss (step_loss): {step_loss:.4f}\n")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"[{timestamp}] Epoch {epoch+1} Step {step+1} | step_loss: {step_loss:.4f}\n")
            log_file.flush()  # Ensures it's written to disk immediately

            iteration += 1 # updated for every STEPS_PER_EPOCH 

        # backpropagate
        epoch_loss = np.mean(meta_batch_losses[-META_BATCH_SIZE*STEPS_PER_EPOCH:])

        # for tracking: print in terminal and save to log file 
        print(f"Epoch {epoch+1}/{EPOCHS}, epoch loss: {epoch_loss:.4f}")
        print ("\n")
        print ("\n")
        log_file.write(f"Epoch {epoch+1}, Epoch Loss: {epoch_loss:.4f}\n")
        log_file.flush()  # Ensures it's written to disk immediately 


        # optionally save checkpoint
        # model.save_weights(os.path.join(SAVE_DIR, f"maml_epoch_{epoch+1}.h5"))

    # final save
    model.save_weights(os.path.join(SAVE_DIR, "final_maml_weights.h5"))
    print("MAML Training Done!")

    # ⏱️ End timer + print summary
    end_time = time.time()
    duration = end_time - start_time

    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"⏱️ Total training time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")   
    log_file.write(f"Total training time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")
    log_file.flush()  

    log_file.close()

if __name__ == "__main__":
    main()
