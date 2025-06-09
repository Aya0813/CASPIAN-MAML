#!/usr/bin/env python
# coding: utf-8
"""
Fine‑tune the previously MAML‑trained CASPIAN model on a *small* set of
SF tasks.  We simply run a few more outer‑loop epochs where the meta‑batch
is sampled **only from the SF TF‑Dataset** you prepared.
"""
import os, tensorflow as tf, numpy as np
from   models import CASPIANv2_MAML as new_models

# ───── Locations ────────────────────────────────────────────────────────────
# BASE_CHECKPOINT = "./models/trained_models/MAML_CASPIAN_patch/final_maml_weights.h5"
# SF_DATASET      = "./data/FSLFSL_MAML_SF_dataset"  # <‑‑ your file
# OUT_DIR         = "./models/trained_models/MAML_CASPIAN_patch_FT_SF"
# os.makedirs(OUT_DIR, exist_ok=True)


# initial code above - bellow just for testing 
# had to re run the part of the data construction code about SF cause the dataset_spec.pb was missing


# BASE_CHECKPOINT = "./models/trained_models/MAML_CASPIAN_patch_latest/final_maml_weights_256.h5"
BASE_CHECKPOINT = "./models/trained_models/MAML_CASPIAN_patch-10epoch-test/final_maml_weights_256.h5"

SF_DATASET      = "./data/SF/FSLFSLFSL_MAML_SF_dataset"  # <‑‑ your file
OUT_DIR         = "./models/trained_models/MAML_CASPIAN_patch_FT_SF_10epoch-test-may22nd"
os.makedirs(OUT_DIR, exist_ok=True)

# ───── Hyper‑params for **fine‑tuning**  (much smaller / shorter) ───────────

# same as the hyperparameters for training with the exception of FINETUNE_EPOCHS
PATCH_SIZE       = 64
SUPPORT_PATCHES  = 10
QUERY_PATCHES    = 10
INNER_STEPS      = 5
INNER_LR         = 0.00002          #  ≈4× smaller than before
OUTER_LR         = 0.000002
META_BATCH_SIZE  = 2 
FINETUNE_EPOCHS  = 25               #  tiny – you only have a few tasks
STEPS_PER_EPOCH  = 50

# ───── Helper to load the TF‑Dataset exactly as before ──────────────────────
def get_sf_mask():
    d = np.load("./sf_full_grid_1024.npy", allow_pickle=True).item()
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

# ───── Patch sampler (unchanged) ────────────────────────────────────────────
def sample_support_query(x_big,y_big,region_mask,slr_val,
                         patch=PATCH_SIZE,supp=SUPPORT_PATCHES,que=QUERY_PATCHES):
    x_np,y_np,mask_np = x_big.numpy(), y_big.numpy(), region_mask.numpy()
    y_min,y_max,eps   = y_np.min(), y_np.max(), 1e-8
    H,W,_ = x_np.shape
    def rand_patch():
        # random top‑left corner
        i0 = np.random.randint(0, H - patch + 1)
        j0 = np.random.randint(0, W - patch + 1)

        # slices (shape: (patch, patch, 1))
        xp = x_np[i0:i0+patch, j0:j0+patch, :]           # ← already 3‑D
        yp = y_np[i0:i0+patch, j0:j0+patch, :]
        mp = mask_np[i0:i0+patch, j0:j0+patch]

        # normalise y
        yp_n = (yp - y_min) / (y_max - y_min + eps)      # (patch,patch,1)

        # build the 4‑channel target
        mp_e   = mp[..., None].astype(np.float32)        # (patch,patch,1)
        y_min_c = np.full_like(yp, y_min, dtype=np.float32)
        y_max_c = np.full_like(yp, y_max, dtype=np.float32)

        y_comb = np.concatenate(
            [yp_n, mp_e, y_min_c, y_max_c], axis=-1)     # (patch,patch,4)

        return xp, y_comb

    x_s, y_s = zip(*(rand_patch() for _ in range(supp)))
    x_q, y_q = zip(*(rand_patch() for _ in range(que)))

    return (np.stack(x_s), np.stack(y_s),
            np.stack(x_q), np.stack(y_q),
            slr_val)


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
ds_sf = load_tf_dataset(SF_DATASET).shuffle(50,reshuffle_each_iteration=True).repeat()
it_sf = iter(ds_sf)

# ───── Fine‑tuning loop (few epochs) ────────────────────────────────────────
for epoch in range(FINETUNE_EPOCHS):
    losses = []
    for step in range(STEPS_PER_EPOCH):
        acc_g, n = None, 0
        for _ in range(META_BATCH_SIZE):
            _,x_big,y_big,mask,slr = next(it_sf)
            xs,ys,xq,yq,slr_v = sample_support_query(
                x_big,y_big,mask,slr,
                patch=PATCH_SIZE,supp=SUPPORT_PATCHES,que=QUERY_PATCHES)
            xs=tf.convert_to_tensor(xs); ys=tf.convert_to_tensor(ys)
            xq=tf.convert_to_tensor(xq); yq=tf.convert_to_tensor(yq)
            slr_t=tf.ones([QUERY_PATCHES,1],tf.float32)*slr_v
            qloss,g = maml.fast(xs,ys,xq,yq,slr_t)
            losses.append(qloss.numpy())
            if acc_g is None: acc_g=g
            else:             acc_g=[a+b for a,b in zip(acc_g,g)]
            n+=1
        acc_g=[g/n for g in acc_g];  maml.meta_update(acc_g)
    print(f"Epoch {epoch+1}/{FINETUNE_EPOCHS}  –  mean query‑loss: {np.mean(losses):.4f}")

# ───── Save fine‑tuned weights ──────────────────────────────────────────────
FT_WEIGHTS = os.path.join(OUT_DIR, "final_maml_ft_sf.h5")
model.save_weights(FT_WEIGHTS)
print("✓  fine‑tuned weights written to", FT_WEIGHTS)
