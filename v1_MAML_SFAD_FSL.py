###### VERSION where support set -> 1 scenario && query set -> 1 scenario


#!/usr/bin/env python
# coding: utf-8
"""
Fine‑tune the previously MAML‑trained CASPIAN model on a *small* set of
SF tasks.  We simply run a few more outer‑loop epochs where the meta‑batch
is sampled **only from the SF TF‑Dataset** you prepared.
"""
import os, tensorflow as tf, numpy as np
from   models import CASPIANv2_MAML as new_models
from datetime import datetime
import time

# ───── Locations ────────────────────────────────────────────────────────────
# og 
# BASE_CHECKPOINT = "./models/trained_models/MAML_CASPIAN_patch/final_maml_weights.h5"
# SF_DATASET      = "./data/FSLFSL_MAML_SF_dataset"  # <‑‑ your file
# OUT_DIR         = "./models/trained_models/MAML_CASPIAN_patch_FT_SF"
# os.makedirs(OUT_DIR, exist_ok=True)
# had to re run the part of the data construction code about SF cause the dataset_spec.pb was missing

### 
# part 1 to change before running the code

MODEL_NAME = "v1_test4"

BASE_CHECKPOINT = f"./models/trained_models/aya/{MODEL_NAME}/final_maml_weights.h5"
# SF_DATASET    = "./data/SF_slr_1m/FSL_MAML_SF_dataset"   # was using the full SF data 
SF_DATASET      = "./data/SF_slr_1m/FSL_MAML_SF_train_ds_aug_split_1"   # now only using a section for training and the rest of teh data val & test 


OUT_DIR         = f"./models/trained_models/aya/{MODEL_NAME}/SF_FT"
log_file_path   = f"{OUT_DIR}/loss_log.txt" 

os.makedirs(OUT_DIR, exist_ok=True)

# ───── Hyper‑params for **fine‑tuning**  (much smaller / shorter) ───────────
# part 2 (Final) to change before running the code
# same as the hyperparameters for training with the exception of FINETUNE_EPOCHS
PATCH_SIZE       = 1024               # same as main training code 
INNER_STEPS      = 5                  # (from 20 in training code) 
INNER_LR         = 0.000002           #  ≈4× smaller than in the main training code
OUTER_LR         = 0.0000002           # also smaller 
META_BATCH_SIZE  = 5                  # same as main training code 

FINETUNE_EPOCHS  = 20                 # set similar to the og code as an initial run 
STEPS_PER_EPOCH  = 10                 # same as main training code 

# ───── Helper to load the TF‑Dataset exactly as before ──────────────────────
def get_sf_mask():
    d = np.load("./data/mask/sf_full_grid_1024.npy", allow_pickle=True).item()
    m  = np.zeros((1024,1024), bool)
    idx= np.array(list(d.values()));  m[idx[:,0],idx[:,1]] = True
    return m.T
MASK_SF = get_sf_mask()

def clear_ds(f,x,y,yf):
    slr_val = 1.0
    m_      = tf.strings.regex_full_match(f, ".*_(\\d+(\\.\\d+)?)m.*")
    slr_val = tf.cond(m_,
                      lambda: tf.strings.to_number(tf.strings.regex_replace(f,".*_(\\d+(\\.\\d+)?)m.*","\\1"), tf.float32),
                      lambda: 1.0)
    return tf.expand_dims(x,-1), tf.expand_dims(y,-1), MASK_SF, slr_val

def map_fn(f,x,y,yf):
    x,y,mask,slr = tf.py_function(clear_ds,[f,x,y,yf],
                                  [tf.float32,tf.float32,tf.bool,tf.float32])
    x.set_shape([1024,1024,1]); y.set_shape([1024,1024,1])
    return f,x,y,mask,slr

def load_tf_dataset(path):
    return tf.data.Dataset.load(path).map(map_fn)


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

# ───── SF fine‑tune dataset iterator (shuffled & repeated) ───────────────────

# be careful -> Since we are trying to get 2 by 2 elements -> if the dataset has an odd number of samples, the final batch will only contain 1 sample
# for this reason we are using drop_remainder=True as without that it was crashing
# ds_sf = load_tf_dataset(SF_DATASET).shuffle(50,reshuffle_each_iteration=True).batch(2, drop_remainder=True).repeat() # og line 
## new line for fasting data handling 
ds_sf = load_tf_dataset(SF_DATASET) \
           .shuffle(50, reshuffle_each_iteration=True) \
           .batch(2, drop_remainder=True) \
           .repeat() \
           .prefetch(tf.data.AUTOTUNE)

it_sf = iter(ds_sf)

log_file = open(log_file_path, "w")  

# Write experiment header
log_file.write("========================================\n")
log_file.write(f"Data used: {SF_DATASET}\n")
log_file.write(f"Base checkpoint path: {BASE_CHECKPOINT}\n")
log_file.write(f"PATCH_SIZE: {PATCH_SIZE}\n")
log_file.write(f"INNER_STEPS: {INNER_STEPS}\n")
log_file.write(f"INNER_LR: {INNER_LR}\n")
log_file.write(f"OUTER_LR: {OUTER_LR}\n")
log_file.write(f"META_BATCH_SIZE: {META_BATCH_SIZE}\n")
log_file.write(f"FINETUNE_EPOCHS: {FINETUNE_EPOCHS}\n")
log_file.write(f"STEPS_PER_EPOCH: {STEPS_PER_EPOCH}\n")
log_file.write("========================================\n\n")
log_file.flush()

# ───── Fine‑tuning loop (few epochs) ────────────────────────────────────────
for epoch in range(FINETUNE_EPOCHS):
    losses = []
    for step in range(STEPS_PER_EPOCH):
        acc_g, n = None, 0
        for _ in range(META_BATCH_SIZE):

            # 2 scenarios instead of 1 (1 for support set and 1 for query set)
            _, x_big_batch, y_big_batch, region_mask_batch, slr_val_batch = next (it_sf)
            # f_str_batch       -> (2,) ← 1 string per scenario
            # x_big_batch       -> (2, 1024, 1024, 1) and y_big_batch -> (2, 1024, 1024, 1)
            # slr_val_batch     -> (2,) ← 1 float per scenario
            # region_mask_batch -> (2, 1024, 1024) ← batch of bool masks ✅
                                #  tf.Tensor([   2 1024 1024], shape=(3,), dtype=int32)

            # get the support set instance 
            x_s = x_big_batch [0:1, ...]                        # shape: (1,1024,1024,1)
            y_s = y_big_batch [0:1, ...]                        # shape: (1,1024,1024,1)
            region_mask_s = region_mask_batch [0]               # shape: tf.Tensor([1024 1024], shape=(2,), dtype=int32) bool
            slr_val_s = slr_val_batch[0]                        


            # get the query set instance
            x_q = x_big_batch [1:2, ...]                         # shape: (1,1024,1024,1)
            y_q = y_big_batch [1:2, ...]                         # shape: (1,1024,1024,1)
            region_mask_q = region_mask_batch [1]                # shape: (1024, 1024) bool 
            slr_val_q = slr_val_batch[1] 

            
            # y final -> shape: (1, 1024, 1024, 4)
            y_s_final_np = process_label_with_mask(y_s, region_mask_s)
            y_q_final_np = process_label_with_mask(y_q, region_mask_q)

            # convert y from numpy to tensor (x is already a tensor) 
            y_s = tf.convert_to_tensor(y_s_final_np, dtype=tf.float32)
            y_q = tf.convert_to_tensor(y_q_final_np, dtype=tf.float32)

            # x -> (1, 1024, 1024, 1) and y -> (1, 1024, 1024, 4)

            # SLR 
            slr_tensor = tf.reshape(slr_val_s, [1, 1])  # shape: (1,1)

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

    print(f"Epoch {epoch+1}/{FINETUNE_EPOCHS}, mean query-loss: {np.mean(losses):.4f}\n")
    log_file.write(f"Epoch {epoch+1}/{FINETUNE_EPOCHS}, mean query-loss: {np.mean(losses):.4f}\n")
    log_file.flush()  # Ensures it's written to disk immediately

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
