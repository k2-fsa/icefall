import argparse
import os
import sys
from datasets import load_dataset, DatasetDict, Audio
import random
import glob
import re

def create_subset_by_hours(
    full_dataset_path,
    output_base_dir,
    target_train_hours,
    random_seed=42,
    duration_column_name='audio_duration'
):
    random.seed(random_seed)

    output_subset_dir = os.path.join(output_base_dir, f'mls_english_subset_{int(target_train_hours)}h')
    os.makedirs(output_subset_dir, exist_ok=True)
    output_subset_data_dir = os.path.join(output_subset_dir, 'data')
    os.makedirs(output_subset_data_dir, exist_ok=True)

    print(f"Attempting to load full dataset from '{full_dataset_path}' using load_dataset...")

    full_data_dir = os.path.join(full_dataset_path, 'data')
    if not os.path.isdir(full_data_dir):
        print(f"Error: Expected a 'data' subdirectory at '{full_data_dir}' containing parquet files. "
              "Please ensure 'full_dataset_path' points to the root of your MLS English download "
              "(e.g., /path/to/mls_english_downloaded_dir) where 'data' is a direct child.", file=sys.stderr)
        sys.exit(1)

    all_parquet_files = glob.glob(os.path.join(full_data_dir, '*.parquet'))
    if not all_parquet_files:
        print(f"Error: No parquet files found in '{full_data_dir}'.", file=sys.stderr)
        sys.exit(1)

    data_files = {}
    split_pattern = re.compile(r'^(train|dev|test)-\d{5}-of-\d{5}\.parquet$')

    print(f"  Discovering splits from filenames in '{full_data_dir}'...")
    for fpath in all_parquet_files:
        fname = os.path.basename(fpath)
        match = split_pattern.match(fname)
        if match:
            split_name = match.group(1)
            if split_name not in data_files:
                data_files[split_name] = []
            data_files[split_name].append(fpath)
        else:
            print(f"Warning: Skipping unrecognized parquet file: {fname}", file=sys.stderr)

    if not data_files:
        print("Error: No recognized train, dev, or test parquet files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found splits and their parquet files: {list(data_files.keys())}")

    try:
        full_dataset = load_dataset("parquet", data_files=data_files)
    except Exception as e:
        print(f"Error loading dataset from '{full_data_dir}' with load_dataset: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(full_dataset, DatasetDict):
        print("Error: The loaded dataset is not a DatasetDict. Expected a DatasetDict structure.", file=sys.stderr)
        sys.exit(1)

    subset_dataset = DatasetDict()
    total_final_duration_ms = 0

    def get_duration_from_column(example):
        if duration_column_name in example:
            return float(example[duration_column_name]) * 1000
        else:
            print(f"Warning: Duration column '{duration_column_name}' not found in example. Returning 0.", file=sys.stderr)
            return 0

    # --- Handle 'dev' split: Copy directly ---
    if 'dev' in full_dataset:
        dev_split = full_dataset['dev']
        subset_dataset['dev'] = dev_split
        dev_duration_ms = sum(get_duration_from_column(ex) for ex in dev_split)
        total_final_duration_ms += dev_duration_ms
        print(f"Copied 'dev' split directly: {len(dev_split)} samples ({dev_duration_ms / (3600*1000):.2f} hours)")
    else:
        print("Warning: 'dev' split not found in original dataset. Skipping copy.")

    # --- Handle 'test' split: Copy directly ---
    if 'test' in full_dataset:
        test_split = full_dataset['test']
        subset_dataset['test'] = test_split
        test_duration_ms = sum(get_duration_from_column(ex) for ex in test_split)
        total_final_duration_ms += test_duration_ms
        print(f"Copied 'test' split directly: {len(test_split)} samples ({test_duration_ms / (3600*1000):.2f} hours)")
    else:
        print("Warning: 'test' split not found in original dataset. Skipping copy.")

    # --- Handle 'train' split: Sample by target hours (stream processing) ---
    target_train_duration_ms = target_train_hours * 3600 * 1000
    current_train_duration_ms = 0
    train_indices_to_include = [] # Store indices of selected samples

    if 'train' in full_dataset:
        train_split = full_dataset['train']
        print(f"\n  Processing 'train' split to reach approximately {target_train_hours} hours...")

        # Get total number of samples in the train split
        total_train_samples = len(train_split)
        print(f"  Total samples in original train split: {total_train_samples}")

        # Create a list of all indices in the train split
        all_train_indices = list(range(total_train_samples))
        random.shuffle(all_train_indices) # Shuffle the indices

        num_samples_processed = 0
        for original_idx in all_train_indices:
            if current_train_duration_ms >= target_train_duration_ms:
                print(f"  Target train hours reached. Stopping processing.")
                break # Target train hours reached, stop adding samples

            example = train_split[original_idx] # Access sample by original index
            duration_ms = get_duration_from_column(example)

            if duration_ms > 0:
                train_indices_to_include.append(original_idx)
                current_train_duration_ms += duration_ms

            num_samples_processed += 1
            if num_samples_processed % 10000 == 0:
                print(f"  Processed {num_samples_processed} samples. Current train duration: {current_train_duration_ms / (3600*1000):.2f} hours")


        # Select the subset from the original split based on chosen indices
        # Sorting is important here to ensure the resulting subset maintains the original order,
        # which can be useful for debugging or consistent processing down the line.
        selected_indices = sorted(train_indices_to_include)
        subset_train_split = train_split.select(selected_indices)

        # Ensure the 'audio' column is correctly typed as Audio feature before saving
        if "audio" in subset_train_split.features and not isinstance(subset_train_split.features["audio"], Audio):
            sampling_rate = subset_train_split.features["audio"].sampling_rate if isinstance(subset_train_split.features["audio"], Audio) else 16000
            new_features = subset_train_split.features
            new_features["audio"] = Audio(sampling_rate=sampling_rate)
            subset_train_split = subset_train_split.cast(new_features)

        subset_dataset['train'] = subset_train_split
        total_final_duration_ms += current_train_duration_ms
        print(f"  Final 'train' split duration: {current_train_duration_ms / (3600*1000):.2f} hours ({len(subset_train_split)} samples)")
    else:
        print("Warning: 'train' split not found in original dataset. No training data will be created.")

    # --- Handle other splits if any, just copy them ---
    for split_name in full_dataset.keys():
        if split_name not in ['train', 'dev', 'test']:
            print(f"Copying unrecognized split '{split_name}' directly.")
            other_split = full_dataset[split_name]
            subset_dataset[split_name] = other_split
            other_duration_ms = sum(get_duration_from_column(ex) for ex in other_split)
            total_final_duration_ms += other_duration_ms
            print(f"  Copied '{split_name}' split: {len(other_split)} samples ({other_duration_ms / (3600*1000):.2f} hours)")


    final_total_hours = total_final_duration_ms / (3600 * 1000)
    print(f"\nOverall subset dataset duration (train + dev + test + others): {final_total_hours:.2f} hours")

    print(f"Saving subset dataset to '{output_subset_dir}' in Parquet format, matching original 'data' structure...")
    try:
        for split_name, ds_split in subset_dataset.items():
            ds_split.to_parquet(os.path.join(output_subset_data_dir, f'{split_name}.parquet'))
            print(f"  Saved split '{split_name}' to '{output_subset_data_dir}'")

        print(f"Successfully created and saved subset dataset to '{output_subset_dir}'")
    except Exception as e:
        print(f"Error saving subset dataset to '{output_subset_dir}': {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a smaller subset of a downloaded Hugging Face audio dataset. "
                    "Copies 'dev' and 'test' splits, and samples 'train' split to a target duration, "
                    "using pre-existing duration column."
    )
    parser.add_argument(
        "--full-dataset-path",
        type=str,
        required=True,
        help="The local path to the already downloaded Hugging Face dataset. "
             "This should be the root directory containing the 'data' subdirectory "
             "(e.g., /path/to/mls_english_download)."
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        required=True,
        help="The base directory where the new subset dataset(s) will be saved. "
             "A subdirectory 'mls_english_subset_Xh' will be created within it."
    )
    parser.add_argument(
        "--target-train-hours",
        type=float,
        required=True,
        help="The approximate total duration of the 'train' split in hours (e.g., 1000 for 1000 hours)."
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for random number generation to ensure reproducibility (default: 42)."
    )
    parser.add_argument(
        "--duration-column-name",
        type=str,
        default='audio_duration',
        help="The name of the column in the dataset that contains the audio duration (assumed to be in seconds). Default: 'audio_duration'."
    )
    args = parser.parse_args()

    create_subset_by_hours(
        args.full_dataset_path,
        args.output_base_dir,
        args.target_train_hours,
        args.random_seed,
        args.duration_column_name
    )

    output_subset_full_path = os.path.join(args.output_base_dir, f'mls_english_subset_{int(args.target_train_hours)}h')
    output_subset_data_path = os.path.join(output_subset_full_path, 'data')