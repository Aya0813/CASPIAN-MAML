# Importing necessary modules
import pathlib  # For object-oriented file system paths
import tensorflow as tf  # TensorFlow for building and handling datasets
import os  # OS module for interacting with the operating system
import numpy as np  # NumPy for numerical operations and loading .npy files
import shutil  # For file operations like removing directories

# Utility function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):  # Check if directory already exists
        os.makedirs(directory)  # Create the directory if not present

# ----------------------------------------------
# Function to create a TensorFlow dataset from input/output .npy files
# ----------------------------------------------

def createDF(ext="*.input.npy", data='train', prefix='SF_', data_dir=''):
    """
    Creates a TF dataset from .npy files. Each scenario file (.input.npy, .output.npy, .output_flat.npy)
    becomes one element in the resulting dataset.
    """

    def process_path(file_path):
        print ("Current file path:", file_path)             # for tracking

        parts = tf.strings.split(file_path, os.path.sep)    # Split file path into components
        fname = tf.strings.join(parts[:-1], "/")            # Join all but the last part (filename)
        fname = fname.numpy().decode('UTF-8') + "/" + parts[-1].numpy().decode('UTF-8')[:-10]    

        label = np.load(fname + ".output.npy")              # shape (1024, 1024)
        label_flat = np.load(fname + ".output_flat.npy")
        input_arr = np.load(file_path.numpy())              # shape (1024, 1024)

        fname = parts[-1]                                   # e.g. '00000100110010011.input.npy' (note: -1 means last element, -2: l avant dernier... )

        return fname, input_arr, label, label_flat
    

    data_dir = data_dir                                                        
    image_count = len(list(data_dir.glob(ext)))                # Count how many files there is with extension .input.npy 
    
    list_ds = tf.data.Dataset.list_files(str(data_dir/ext), shuffle=False)          # Create a dataset of file paths from data_dir matching ext="*.input.npy"
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)          # Shuffle the file paths once, in a fixed way. - why shuffle? Shuffling ensures that the model sees a random mix of data points across batches, which improves generalization and reduces overfitting.
    ds_paths = list_ds
    print("Dataset length: %d" % tf.data.experimental.cardinality(ds_paths).numpy())


    # Map the path processing function across all dataset paths
    ds = ds_paths.map(
        lambda paths: tf.py_function(
            process_path,
            inp=[paths],
            Tout=[tf.string, tf.float32, tf.float32, tf.float32]  # Output types
        )
    )

    # Choose directory based on training or holdout flag
    savedir = f"./data/FSL{prefix}dataset" if data == 'train' else f"./data/FSL_{prefix}holdout_dataset"
    create_directory(savedir)  # Create the directory if needed
    tf.data.experimental.save(ds, savedir)  # Save the TensorFlow dataset
    print(f"{prefix}DB has been created")  # Confirmation message


# ----------------------------------------------
# Function to apply random region masking ("cutouts") for data augmentation
# ----------------------------------------------


def rnd_rmv(f, x, y, yf, window, version, num_cutouts, scale_factor):
    """
    Performs random cutouts on the grid where x == 1 or x == 0.5.
    Used in your augmentation pipeline.
    """

    # Build new filename based on cutout parameters
    fname = f.numpy().decode('UTF-8') + "_rnd" + str(window.numpy()) + "_v" + str(version.numpy())

    # Create a mask of all cells where x == 1.0 or x == 0.5
    mask = tf.logical_or(tf.math.equal(x, 1.0), tf.math.equal(x, 0.5))

    # Get all coordinates where the mask is True
    arr = np.vstack((np.where(mask == True)[0], np.where(mask == True)[1])).T

    # Create random indices to select cutout locations
    idxs = tf.range(tf.shape(arr)[0])
    ridxs = tf.random.shuffle(idxs)[:num_cutouts]

    # Get random input coordinates to apply cutouts
    rinput = tf.gather(arr, ridxs)
    rinput = rinput.numpy()  # Convert to numpy array
    new_x = x.numpy()  # Convert tensor to numpy array for editing

    # Apply zero masking to defined regions (cutouts)
    for i in range(num_cutouts):
        start_i = max(0, rinput[i, 0] - window)
        end_i = min(new_x.shape[0], rinput[i, 0] + window)
        start_j = max(0, rinput[i, 1] - window)
        end_j = min(new_x.shape[1], rinput[i, 1] + window)
        new_x[start_i:end_i, start_j:end_j] = 0  # Set region to zero

    return fname, new_x, y, yf  # Return modified input and original labels

# ----------------------------------------------
# Function to split dataset and apply data augmentation
# ----------------------------------------------


def split_and_augmentDF(window=10, num_cutouts=1, repeats=1, original_ds=True, scale_factor=1, splits=3, prefix='SF_'):
    """
    Splits the dataset into train/val/test and optionally applies random cutout augmentation.
    """
    # Create necessary directories
    create_directory("./data/temp")
    create_directory(f"./data/FSL_{prefix}dataset")

    for s in range(splits):  # Loop over different data splits
        ds = tf.data.experimental.load(f"./data/FSL{prefix}dataset")  # Load the saved dataset

        # Split dataset sizes
        ds_size = int(ds.cardinality())
        train_size = int(0.79 * ds_size)
        val_size = int((ds_size - train_size) * 0.4)
        test_size = ds_size - val_size - train_size

        ds = ds.shuffle(300, seed=s)  # Shuffle the dataset with split-specific seed

        # Reset temp folder
        if os.path.exists("./data/temp/"):
            shutil.rmtree("./data/temp/")
        tf.data.experimental.save(ds, "./data/temp/")  # Save shuffled dataset to disk
        ds = tf.data.experimental.load("./data/temp/")  # Reload it fresh

        # Slice dataset into train, test, val
        train_dataset = ds.take(train_size)
        test_dataset = ds.skip(train_size).take(test_size)
        val_dataset = ds.skip(train_size + test_size)

        # Log dataset lengths
        print("Training dataset length: %d" % tf.data.experimental.cardinality(train_dataset).numpy())
        print("Validation dataset length: %d" % tf.data.experimental.cardinality(val_dataset).numpy())
        print("Testing dataset length: %d" % tf.data.experimental.cardinality(test_dataset).numpy())

        # Save datasets
        print(f"Saving the {prefix} test datasets ....")
        tf.data.experimental.save(test_dataset, f"./data/FSL_{prefix}test_ds_split_{s+1}")
        tf.data.experimental.save(train_dataset, f"./data/FSL_{prefix}train_ds_split_{s+1}")
        tf.data.experimental.save(val_dataset, f"./data/FSL_{prefix}val_ds_split_{s+1}")
        print("Done saving, proceeding to augmentation....")

        # Augmentation for train and val sets
        for k in range(2):  # 0 = train, 1 = val
            temp_ds = train_dataset if k == 0 else val_dataset
            dss = []
            if original_ds:  # Add unaugmented dataset
                dss.append(temp_ds)

            for i in range(repeats):  # Repeat augmentation
                print(f"Augmented DB slice {i} is in creation.......")
                ds_augmented = temp_ds.map(
                    lambda f, x, y, yf: tf.py_function(
                        rnd_rmv,
                        inp=[f, x, y, yf, window, i, num_cutouts, scale_factor],
                        Tout=[tf.string, tf.float32, tf.float32, tf.float32]
                    )
                )
                dss.append(ds_augmented)

            # Combine original + augmented datasets
            combined_dataset = dss[0]
            for i in range(1, len(dss)):
                combined_dataset = combined_dataset.concatenate(dss[i])

            combined_dataset = combined_dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch for performance

            # Save final augmented dataset
            ds_type = "train_ds_aug" if k == 0 else "val_ds_aug"
            name = f"./data/FSL_{prefix}{ds_type}_split_{s+1}"
            create_directory(name)
            tf.data.experimental.save(combined_dataset, name)
        print(f"{prefix}Augmented DBs have been created")

    # Clean up temp folder
    if os.path.exists("./data/temp/"):
        shutil.rmtree("./data/temp/")


# ----------------------------------------------
# Function to combine two dataset prefixes into a combined one
# ----------------------------------------------

def combine_datasets():
    """
    Combines SF_ and AD_ datasets into a single Combined_ dataset if desired.
    """
    def combine_and_save(prefix1, prefix2, combined_prefix, ds_type, split_num):
        ds1 = tf.data.experimental.load(f"./data/FSL_{prefix1}{ds_type}_split_{split_num}")
        ds2 = tf.data.experimental.load(f"./data/FSL_{prefix2}{ds_type}_split_{split_num}")
        combined_ds = ds1.concatenate(ds2)
        combined_ds = combined_ds.prefetch(tf.data.AUTOTUNE)
        create_directory(f"./data/FSL_{combined_prefix}{ds_type}_split_{split_num}")
        tf.data.experimental.save(combined_ds, f"./data/FSL_{combined_prefix}{ds_type}_split_{split_num}")
    
    splits = 1   # Number of dataset splits to combine
    for s in range(splits):
        for ds_type in ["train_ds", "val_ds", "test_ds", "train_ds_aug", "val_ds_aug"]:
            print(f"Combining datasets for {ds_type} split {s+1}")
            combine_and_save('SF_', 'AD_', 'Combined_', ds_type, s+1)
            print(f"Combined datasets for {ds_type} split {s+1} have been created")



##################################################
#  1) Create AD dataset for Main Training
##################################################

# createDF(ext="*.input.npy", 
#          data='train', 
#          prefix='MAML_AD_', 
#          data_dir=pathlib.Path("./data/Training_Validation_Testing_AD"))

# split_and_augmentDF(window=30, 
#                     repeats=23, 
#                     original_ds=True, 
#                     num_cutouts=2, 
#                     splits=1, 
#                     prefix='MAML_AD_')

##################################################
#  2) Create SF dataset
##################################################
# createDF(ext="*.input.npy", 
#          data='train', 
#          prefix='FSL2_AD_', 
#          data_dir=pathlib.Path("./data/FSLFSL_MAML_SF_dataset"))
    # gives you 2 files one for the MAML training and one for the FT 
    # why MAML then FT 
            # initial main training - with AD
            # and then SF for fine tuning on SF 
    # why use split 1 for the MAML training code
            # why different splits - to show the model diff. cases 
            # bilal - change it to only 1 split 
            # you can remove the split thing if you want

# initially the code above but had to change the last line cause wrong folder naming - SF instead of AD  
createDF(ext="*.input.npy", 
         data='train', 
         prefix='FSLFSL_MAML_SF_', 
         data_dir=pathlib.Path("./data/FSLFSL_MAML_SF_dataset"))



# dataset_spec.pb is a metadata file automatically created by TensorFlow when you save a dataset 




### meeting with Bilal - May 16th 
    ### FSL - meaning Few Shot Learning 
    # trianig  val test - complete AD data
