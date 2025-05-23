#!/usr/bin/env python
# coding: utf-8
"""
Evaluate the MAML‑trained CASPIAN model on the AD / SF *test* scenarios.

The script:
1.  loads each 1024×1024 scenario grid,
2.  splits it into non‑overlapping 64×64 patches (batch inference),
3.  re‑assembles the predicted patches back into a full‑size grid,
4.  computes the same metrics you used before,
5.  writes predictions + Grad‑CAM overlays to disk,
6.  exports the metrics to an Excel workbook.
"""
import os, re, glob
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ──────────────────────────────────────────────────────────────────────────────
#  Project‑specific imports
# ──────────────────────────────────────────────────────────────────────────────
from models import CASPIANv2_MAML as caspian_models
from models.keras_vision_transformer import transformer_layers, swin_layers
from utilsSFAD import artae_score, delta_score, acc0   # <- already in your repo

# ──────────────────────────────────────────────────────────────────────────────
#  Paths / constants
# ──────────────────────────────────────────────────────────────────────────────
# PATCH_SIZE       = 256
# MODEL_WEIGHTS    = "./models/trained_models/MAML_CASPIAN_patch/final_maml_weights_256.h5"
# EVAL_DATASET     = "C:/NYUAD/CASPIAN/AregSF/data/FSL_MAML_AD_test_ds_split_1"


# above code is the original - bellow just for initial testing 
PATCH_SIZE       = 64
#MODEL_WEIGHTS    = "./models/trained_models/MAML_CASPIAN_patch_FT_SF_10epoch-test-final-step/final_maml_ft_sf.h5"
MODEL_WEIGHTS    = "./models/trained_models/MAML_CASPIAN_patch/final_maml_weights_256.h5"
EVAL_DATASET     = "./data/AD/test/FSL_MAML_AD_test_ds_split_1"


### 

SAVE_PRED_DIR       = "./Results/MAML-May22nd-og/Predictions"
SAVE_EXPL_IMG_DIR   = "./Results/MAML-May22nd-og/Explainability"
SAVE_EXPL_NPY_DIR   = "./Results/MAML-May22nd-og/Explainability NPY"
os.makedirs(SAVE_PRED_DIR,      exist_ok=True)
os.makedirs(SAVE_EXPL_IMG_DIR,  exist_ok=True)
os.makedirs(SAVE_EXPL_NPY_DIR,  exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ──────────────────────────────────────────────────────────────────────────────
def normalize(a):
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn + 1e-8)

def make_gradcam_heatmap(inputs, model, last_layer="activation_12"):
    """
    inputs = {"input_grid": patch[None,...], "slr_input": slr_val[None,1]}
    returns heatmap as 2‑D numpy array in [0,1]
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inputs, training=False)
        class_channel   = preds[...,0]                           # regression
    grads     = tape.gradient(class_channel, conv_out)[0]        # (H,W,C)
    weights   = tf.reduce_mean(grads, axis=(0,1))                # (C,)
    cam       = tf.reduce_sum(weights * conv_out[0], axis=-1)    # (H,W)
    cam       = np.maximum(cam, 0)
    cam       = normalize(cam)
    cam       = tf.image.resize(cam[...,None],
                                (PATCH_SIZE, PATCH_SIZE)).numpy()[...,0]
    return cam

def save_gradcam_overlay(img_patch, heatmap, save_png, save_npy, alpha=0.5):
    """
    img_patch : single‑channel patch (64×64)
    heatmap   : 64×64 in [0,1]
    Saves overlay PNG + raw .npy heatmap.
    """
    img = np.squeeze(img_patch)
    mask_1  = img == 1
    mask_05 = img == .5

    jet = plt.get_cmap("jet")(np.arange(256))[:,:3]
    hmap = (heatmap*255).astype('uint8')
    hmap_rgb = jet[hmap]

    # everything not belonging to a mask becomes the darkest colormap colour
    hmap_rgb[~(mask_1 | mask_05)] = jet[0]

    overlay = normalize((1-alpha)*np.ones((*img.shape,3)) + alpha*hmap_rgb)
    overlay = np.flipud(overlay)

    plt.imsave(save_png, overlay)
    np.save(save_npy, heatmap)

# ──────────────────────────────────────────────────────────────────────────────
#  Patch <-> grid helpers
# ──────────────────────────────────────────────────────────────────────────────
def grid_to_patches(grid, patch=PATCH_SIZE):
    """(H,W,1) -> (num_patches, patch, patch, 1) in row‑major order"""
    h, w, _ = grid.shape
    patches = []
    for i in range(0, h, patch):
        for j in range(0, w, patch):
            patches.append(grid[i:i+patch, j:j+patch, :])
    return np.stack(patches, 0)                                  # (N,64,64,1)

def patches_to_grid(patches, patch=PATCH_SIZE):
    """inverse of grid_to_patches (assumes non‑overlap 1024/patch divisible)"""
    num = patches.shape[0]
    per_row = 1024 // patch
    rows = []
    for r in range(per_row):
        row = np.concatenate(patches[r*per_row:(r+1)*per_row], axis=1)
        rows.append(row)
    return np.concatenate(rows, axis=0)                          # (1024,1024,1)

# ──────────────────────────────────────────────────────────────────────────────
#  Load masks (same routine you already have)
# ──────────────────────────────────────────────────────────────────────────────
def get_sf_mask():
    d = np.load("./sf_full_grid_1024.npy", allow_pickle=True).item()
    m = np.zeros((1024,1024), bool)
    idx = np.array(list(d.values()))
    m[idx[:,0], idx[:,1]] = True
    return m.T
def get_ad_mask():
    d = np.load("./ad_full_grid_1024.npy", allow_pickle=True).item()
    m = np.zeros((1024,1024), bool)
    idx = np.array(list(d.values()))
    m[idx[:,0], idx[:,1]] = True
    return m.T
MASK_SF = get_sf_mask()
MASK_AD = get_ad_mask()

# ──────────────────────────────────────────────────────────────────────────────
#  Build CASPIAN and load MAML weights
# ──────────────────────────────────────────────────────────────────────────────
model = caspian_models.CASPIAN(
    input_shape=(PATCH_SIZE, PATCH_SIZE, 1),
    filters=72, input_centering=True,
    depth=4, cardinality=34, activation='tanh',
    bottleneck_depth=8, init="glorot_normal",
    sup_level=1, compression_factor=0.85,
    gr_level=4, slr_input=True
)
# build once
_ = model({"input_grid": tf.zeros([1,PATCH_SIZE,PATCH_SIZE,1]),
           "slr_input": tf.zeros([1,1])})
model.load_weights(MODEL_WEIGHTS)
print("✓  MAML weights loaded")

# ──────────────────────────────────────────────────────────────────────────────
#  Prepare evaluation TF‑dataset (unchanged – uses your hold‑out *.save)
# ──────────────────────────────────────────────────────────────────────────────
def clear_ds_eval(f,x,y,yf):
    """Produce (scenario_str, full_input_grid, full_label_grid, mask, slr)"""
    scenario = f.numpy().decode()
    m = MASK_SF if len(scenario.split('.')[0])==30 else MASK_AD
    # get SLR
    m_ = re.search(r'_(\d+(\.\d+)?)m', scenario)
    slr_val = float(m_.group(1)) if m_ else 1.0
    return scenario, tf.expand_dims(x,-1), tf.expand_dims(y,-1), m, slr_val

def map_eval(f,x,y,yf):
    return tf.py_function(
        clear_ds_eval, inp=[f,x,y,yf],
        Tout=[tf.string, tf.float32, tf.float32, tf.bool, tf.float32]
    )

ds_eval = tf.data.Dataset.load(EVAL_DATASET).map(map_eval).batch(1)

# ──────────────────────────────────────────────────────────────────────────────
#  Metric containers
# ──────────────────────────────────────────────────────────────────────────────
metrics = {
    'sf':  {'mae':[], 'rmse':[], 'r2':[], 'rtae':[], 'd05':[], 'd01':[], 'acc':[]},
    'ad':  {'mae':[], 'rmse':[], 'r2':[], 'rtae':[], 'd05':[], 'd01':[], 'acc':[]},
    'all': {'mae':[], 'rmse':[], 'r2':[], 'rtae':[], 'd05':[], 'd01':[], 'acc':[]}
}

# ──────────────────────────────────────────────────────────────────────────────
#  Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────
# ─── inside the evaluation loop ──────────────────────────────────────────
for scenario_b, x_full_b, y_full_b, mask_b, slr_b in ds_eval:
    # scenario_b.shape == (1,) and dtype == 'S' (byte‑string)
    scenario = scenario_b.numpy()[0].decode("utf‑8")   # <-- fixed line
    x_full     = x_full_b.numpy()[0]        # (1024,1024,1)
    y_full     = y_full_b.numpy()[0]        # (1024,1024,1)
    mask_bool  = mask_b.numpy()[0]          # (1024,1024)  ← squeeze the batch‑dim
    slr_val    = slr_b.numpy()

    # Split into patches & batch‑predict
    patches   = grid_to_patches(x_full)
    slr_batch = np.ones((patches.shape[0],1), np.float32)*slr_val
    preds_p   = model.predict({"input_grid":patches, "slr_input":slr_batch},
                              verbose=0)
    pred_grid = patches_to_grid(preds_p)[:,:,0]       # (1024,1024)

    # Prepare label channels for un‑normalisation
    # ─── extract 2‑D arrays and mask them ──────────────────────────────────
    y_true_2d = y_full[:,:,0]          # raw ground‑truth depths
    y_pred_2d = pred_grid              # model prediction (already raw)

    valid_t = y_true_2d[mask_bool]     # 1‑D vectors over inundation mask
    valid_p = y_pred_2d[mask_bool]


    # Store metrics
    def update(dic):
        dic['mae'].append( mean_absolute_error(valid_t, valid_p) )
        dic['rmse'].append(mean_squared_error(valid_t, valid_p))
        dic['r2'].append(  r2_score        (valid_t, valid_p) )
        dic['rtae'].append(artae_score(np.array([valid_t]), np.array([valid_p])))
        dic['d05'].append( delta_score(np.array([valid_t]), np.array([valid_p]),0.5).mean())
        dic['d01'].append( delta_score(np.array([valid_t]), np.array([valid_p]),0.1).mean())
        dic['acc'].append(  acc0(np.array([valid_t]), np.array([valid_p])).mean())
    region = 'sf' if len(scenario.split('.')[0])==30 else 'ad'
    update(metrics[region]);  update(metrics['all'])

    # ─── Save prediction / label ────────────────────────────────────────────
    base = os.path.splitext(os.path.basename(scenario))[0]
    np.save(os.path.join(SAVE_PRED_DIR,  f"{base}_pred.npy"),  y_pred_2d)
    np.save(os.path.join(SAVE_PRED_DIR,  f"{base}_label.npy"), y_true_2d)

    # ---------- NEW : also save as CSV -----------------------------------
    np.savetxt(os.path.join(SAVE_PRED_DIR, f"{base}_pred.csv"),
               y_pred_2d,  delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(SAVE_PRED_DIR, f"{base}_label.csv"),
               y_true_2d,  delimiter=",", fmt="%.6f")
    # ---------------------------------------------------------------------


    # ---------- NEW : also dump them as colour‑mapped PNGs ------------------
    def save_png(arr, path, cmap="viridis"):
        plt.imsave(path, normalize(arr), cmap=cmap)

    save_png(x_full[:,:,0] ,          # input grid (0/0.5/1)
             os.path.join(SAVE_PRED_DIR, f"{base}_input.png"),
             cmap="gray")

    save_png(y_pred_2d,
             os.path.join(SAVE_PRED_DIR, f"{base}_pred.png"))

    save_png(y_true_2d,
             os.path.join(SAVE_PRED_DIR, f"{base}_label.png"))
    # ------------------------------------------------------------------------

    # ─── Grad‑CAM for the *centre patch* (quick visual check) ───────────────
    centre_idx = ( (1024//PATCH_SIZE)//2 )   # middle patch row / col
    centre_patch = patches[centre_idx*(1024//PATCH_SIZE)+centre_idx] # (64,64,1)
    cam = make_gradcam_heatmap(
        {"input_grid":centre_patch[None],
         "slr_input": np.array([[slr_val]],dtype=np.float32)},
        model
    )
    save_gradcam_overlay(
        centre_patch, cam,
        os.path.join(SAVE_EXPL_IMG_DIR,  f"{base}_gradcam.png"),
        os.path.join(SAVE_EXPL_NPY_DIR,  f"{base}_gradcam.npy")
    )

# ──────────────────────────────────────────────────────────────────────────────
#  Aggregate & export metrics
# ──────────────────────────────────────────────────────────────────────────────
def summarise(dic):
    return [f"{np.mean(dic[k]):.5f} ± {np.std(dic[k]):.5f}" for k in
            ['mae','rmse','rtae','d05','d01','r2','acc']]

summary = pd.DataFrame({
    "Metric": ["MAE","RMSE","RTAE","δ0.5","δ0.1","R²","Acc0"],
    "SF":     summarise(metrics['sf']),
    "AD":     summarise(metrics['ad']),
    "All":    summarise(metrics['all'])
})
print("\n\n", summary, "\n")

summary.to_excel("./Results/MAML_Eval.xlsx", index=False)
print("✓  Results written to Results/MAML_Eval.xlsx")
