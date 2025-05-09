#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import tensorflow as tf
import numpy as np
import keras
import re
from datetime import datetime
import time

# Import your CASPIAN model
from models import CASPIANv2_MAML as new_models

################################################################################
#                              MAML HYPERPARAMS
################################################################################
PATCH_SIZE = 256        # dimension of each patch: 64x64
SUPPORT_PATCHES = 20    # how many patches in the support set
QUERY_PATCHES   = 20    # how many patches in the query set
INNER_STEPS = 10        # steps in the MAML inner loop
INNER_LR = 0.00085      # learning rate for inner loop
OUTER_LR = 0.000085     # learning rate for outer loop
META_BATCH_SIZE = 10    # how many tasks (scenarios) in each meta-batch
EPOCHS = 200            # how many outer-loop epochs
STEPS_PER_EPOCH = 100   # how many meta-batches per epoch

MODEL_NAME = "MAML_CASPIAN_patch"
SAVE_DIR = f"./models/trained_models/{MODEL_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)

################################################################################
#                     LOAD & PREPARE EACH SCENARIO
################################################################################

def get_sf_mask():
    mask = np.load("./sf_full_grid_1024.npy", allow_pickle=True).item()
    the_mask = np.zeros((1024, 1024), dtype=bool)
    idx = np.array(list(mask.values()))
    the_mask[idx[:, 0], idx[:, 1]] = True
    return the_mask.T

def get_ad_mask():
    mask = np.load("./ad_full_grid_1024.npy", allow_pickle=True).item()
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
    f_str = f.numpy().decode('UTF-8')
    slr_val = 1.0
    match = re.search(r'_(\d+(\.\d+)?)m', f_str)
    if match:
        slr_val = float(match.group(1))

    # Identify region
    if len(f_str.split('_')[0]) == 30:
        region_mask = sf_mask
    else:
        region_mask = ad_mask

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
#                          PATCH-BASED SAMPLING
################################################################################

def sample_support_and_query_patches(x_big, y_big, region_mask, slr_val, 
                                     patch_size=64, 
                                     support_count=4, 
                                     query_count=4):
    """
    - x_big, y_big:  (1024,1024,1) from the scenario
    - region_mask:   (1024,1024) bool
    - slr_val:       float for the entire scenario
    - patch_size:    e.g. 64
    - support_count, query_count: number of patches for support/query sets
    Returns:
      x_support: (support_count, patch_size, patch_size, 1)
      y_support: (support_count, patch_size, patch_size, 4)  ( [y_norm, mask, y_min, y_max] )
      x_query  : (query_count, patch_size, patch_size, 1)
      y_query  : (query_count, patch_size, patch_size, 4)
      slr_val  : same float
    """
    # Convert to numpy for random sampling
    x_np = x_big.numpy()
    y_np = y_big.numpy()
    mask_np = region_mask.numpy()

    # Compute global y_min, y_max for entire scenario => consistent normalization
    y_flat = y_np.reshape(-1)
    y_min = float(y_flat.min())
    y_max = float(y_flat.max())
    eps = 1e-8

    # We'll define a helper to sample a single patch
    H, W, _ = x_np.shape

    def sample_random_patch():
        # pick top-left corner
        max_i = H - patch_size
        max_j = W - patch_size
        i0 = np.random.randint(0, max_i+1)
        j0 = np.random.randint(0, max_j+1)
        # slice
        x_patch = x_np[i0:i0+patch_size, j0:j0+patch_size, :]
        y_patch = y_np[i0:i0+patch_size, j0:j0+patch_size, :]
        m_patch = mask_np[i0:i0+patch_size, j0:j0+patch_size]
        # normalize y_patch
        y_patch_norm = (y_patch - y_min)/(y_max - y_min + eps)
        # build [y_norm, mask, y_min, y_max]
        # shape => (patch_size, patch_size, 4)
        m_patch_expanded = np.expand_dims(m_patch, axis=-1).astype(np.float32)
        y_min_patch = np.full_like(y_patch_norm, y_min, dtype=np.float32)
        y_max_patch = np.full_like(y_patch_norm, y_max, dtype=np.float32)
        y_combined = np.concatenate([y_patch_norm, m_patch_expanded, y_min_patch, y_max_patch], axis=-1)
        return x_patch, y_combined

    # sample support_count patches
    x_support_list = []
    y_support_list = []
    for _ in range(support_count):
        xp, yp = sample_random_patch()
        x_support_list.append(xp)
        y_support_list.append(yp)
    
    # sample query_count patches
    x_query_list = []
    y_query_list = []
    for _ in range(query_count):
        xp, yp = sample_random_patch()
        x_query_list.append(xp)
        y_query_list.append(yp)

    # stack => shape (support_count, patch_size, patch_size, ?)
    x_support = np.stack(x_support_list, axis=0)
    y_support = np.stack(y_support_list, axis=0)
    x_query   = np.stack(x_query_list, axis=0)
    y_query   = np.stack(y_query_list, axis=0)

    return x_support, y_support, x_query, y_query, slr_val


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
    y_norm  = y_true[:,:,:,0]
    mask    = y_true[:,:,:,1]
    y_min   = y_true[:,:,:,2]
    y_max   = y_true[:,:,:,3]

    # Pred has shape (batch, patch_size, patch_size, 1)
    y_pred  = tf.squeeze(y_pred, axis=-1)

    # unnormalize
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
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", keras.__version__)
    print("GPU available:", tf.test.is_gpu_available())

    # 1) Load your CASPIAN model
    model = new_models.CASPIAN(
        input_shape=(PATCH_SIZE, PATCH_SIZE, 1),  # 64x64 patches
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

    # 3) Load your dataset. Example: combined train set
    ds_train = load_tf_dataset("./data/FSL_MAML_AD_train_ds_aug_split_1")
    ds_train = ds_train.shuffle(200, reshuffle_each_iteration=True).repeat()

    # We'll iterate in Python, pulling one scenario at a time
    ds_iter = iter(ds_train)

    iteration = 0
    for epoch in range(EPOCHS):
        meta_batch_losses = []

        for step in range(STEPS_PER_EPOCH):
            # accumulate grads across tasks in a meta-batch
            grads_accum = None
            task_count = 0

            for _ in range(META_BATCH_SIZE):
                f_str, x_big, y_big, region_mask, slr_val = next(ds_iter)
                # sample patches for support/query
                x_s_np, y_s_np, x_q_np, y_q_np, scenario_slr = sample_support_and_query_patches(
                    x_big, y_big, region_mask, slr_val,
                    patch_size=PATCH_SIZE,
                    support_count=SUPPORT_PATCHES,
                    query_count=QUERY_PATCHES
                )
                # Convert to TF
                # shape => (support_count, 64,64,1)
                x_s = tf.convert_to_tensor(x_s_np, dtype=tf.float32)
                y_s = tf.convert_to_tensor(y_s_np, dtype=tf.float32)
                x_q = tf.convert_to_tensor(x_q_np, dtype=tf.float32)
                y_q = tf.convert_to_tensor(y_q_np, dtype=tf.float32)
                # SLR => shape (support_count, 1) or (1,1). We'll do (support_count,1) so each patch is a batch item
                # But we can also do just (1,1) broadcast. Either works.
                slr_tensor_supp = tf.ones([SUPPORT_PATCHES,1], dtype=tf.float32)*scenario_slr
                slr_tensor_query= tf.ones([QUERY_PATCHES,1], dtype=tf.float32)*scenario_slr

                # fast adapt
                q_loss, grads_query = maml.fast_adapt(x_s, y_s, x_q, y_q, slr_tensor_query)
                meta_batch_losses.append(q_loss.numpy())

                if grads_accum is None:
                    grads_accum = grads_query
                else:
                    grads_accum = [ga + g for ga, g in zip(grads_accum, grads_query)]
                task_count += 1

            # after one meta-batch of tasks, average the grads & update
            grads_accum = [g / task_count for g in grads_accum]
            maml.meta_update(grads_accum)
            iteration += 1

        epoch_loss = np.mean(meta_batch_losses[-META_BATCH_SIZE*STEPS_PER_EPOCH:])
        print(f"Epoch {epoch+1}/{EPOCHS}, iteration {iteration}, query loss: {epoch_loss:.4f}")

        # optionally save checkpoint
        # model.save_weights(os.path.join(SAVE_DIR, f"maml_epoch_{epoch+1}.h5"))

    # final save
    model.save_weights(os.path.join(SAVE_DIR, "final_maml_weights_256.h5"))
    print("MAML Training Done!")

if __name__ == "__main__":
    main()
