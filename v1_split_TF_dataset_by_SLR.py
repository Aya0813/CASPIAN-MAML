
# For running the v1 code:  
# Not currently in use - this will be needed if the dataset includes multiple SLR values per region.

import tensorflow as tf
import os
from collections import defaultdict

# Reuse your existing functions
from your_module import load_tf_dataset  # or paste in clear_ds + map_fn + load_tf_dataset

# Path to original full dataset
DATASET_PATH = "./data/AD/train/FSL_MAML_AD_train_ds_aug_split_1"
SAVE_BASE_PATH = "./data/AD/train/split_by_slr"

# Load full dataset
full_ds = load_tf_dataset(DATASET_PATH)

# Convert dataset to a dictionary of slr: list of elements
slr_to_examples = defaultdict(list)

print("Grouping scenarios by SLR...")
for example in full_ds:
    f_str, x, y, mask, slr_val = example
    slr_val = float(slr_val.numpy())  # convert tensor to float
    slr_to_examples[slr_val].append(example)

print(f"Found scenarios for SLR values: {list(slr_to_examples.keys())}")

# Now save each SLR group to a separate dataset
for slr_val, examples in slr_to_examples.items():
    slr_dir = os.path.join(SAVE_BASE_PATH, f"SLR_{slr_val:.1f}")
    os.makedirs(slr_dir, exist_ok=True)

    # Convert to TensorFlow Dataset
    ds_slr = tf.data.Dataset.from_generator(
        lambda: (ex for ex in examples),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),        # f_str
            tf.TensorSpec(shape=(1024, 1024, 1), dtype=tf.float32),  # x
            tf.TensorSpec(shape=(1024, 1024, 1), dtype=tf.float32),  # y
            tf.TensorSpec(shape=(1024, 1024), dtype=tf.bool),        # mask
            tf.TensorSpec(shape=(), dtype=tf.float32)         # slr_val
        )
    )

    # Save the dataset
    tf.data.experimental.save(ds_slr, slr_dir)
    print(f"✓ Saved {len(examples)} scenarios to {slr_dir}")


# then in MAML_SFAD_Training.py we can do direct:
# ds_train = load_tf_dataset("./data/AD/train/split_by_slr/SLR_0.5")
# ds_train = ds_train.shuffle(...).batch(2).repeat()
