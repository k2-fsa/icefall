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
    target_dev_hours,  # New parameter
    target_test_hours, # New parameter
    random_seed=42,
    duration_column_name='audio_duration'
):
    random.seed(random_seed)

    output_subset_dir = os.path.join(output_base_dir, f'mls_english_subset_train{int(target_train_hours)}h_dev{int(target_dev_hours)}h_test{int(target_test_hours)}h')
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
    # Expanded pattern to also detect 'validation' if it's in filenames
    split_pattern = re.compile(r'^(train|dev|test|validation)-\d{5}-of-\d{5}\.parquet$')

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
        print("Error: No recognized train, dev, test, or validation parquet files found.", file=sys.stderr)
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

    # --- Renaming 'validation' split to 'dev' if necessary ---
    if 'validation' in full_dataset:
        if 'dev' in full_dataset:
            print("Warning: Both 'dev' and 'validation' splits found in the original dataset. Keeping 'dev' and skipping rename of 'validation'.", file=sys.stderr)
        else:
            print("Renaming 'validation' split to 'dev' for consistent keying.")
            full_dataset['dev'] = full_dataset.pop('validation')
    # --- End Renaming ---

    subset_dataset = DatasetDict()
    total_final_duration_ms = 0

    def get_duration_from_column(example):
        """Helper to safely get duration from the specified column, in milliseconds."""
        if duration_column_name in example:
            return float(example[duration_column_name]) * 1000
        else:
            print(f"Warning: Duration column '{duration_column_name}' not found in example. Returning 0.", file=sys.stderr)
            return 0

    # --- NEW: Generalized sampling function ---
    def sample_split_by_hours(split_name, original_split, target_hours):
        """
        Samples a dataset split to reach a target number of hours.
        Returns the sampled Dataset object and its actual duration in milliseconds.
        """
        target_duration_ms = target_hours * 3600 * 1000
        current_duration_ms = 0
        indices_to_include = []

        if original_split is None or len(original_split) == 0:
            print(f"  Warning: Original '{split_name}' split is empty or not found. Cannot sample.", file=sys.stderr)
            return None, 0

        print(f"\n  Processing '{split_name}' split to reach approximately {target_hours} hours...")
        print(f"  Total samples in original '{split_name}' split: {len(original_split)}")

        all_original_indices = list(range(len(original_split)))
        random.shuffle(all_original_indices) # Shuffle indices for random sampling

        num_samples_processed = 0
        for original_idx in all_original_indices:
            if current_duration_ms >= target_duration_ms and target_hours > 0:
                print(f"  Target {split_name} hours reached ({target_hours}h). Stopping processing.")
                break

            example = original_split[original_idx]
            duration_ms = get_duration_from_column(example)

            if duration_ms > 0:
                indices_to_include.append(original_idx)
                current_duration_ms += duration_ms
            
            num_samples_processed += 1
            if num_samples_processed % 10000 == 0: # Print progress periodically
                print(f"  Processed {num_samples_processed} samples for '{split_name}'. Current duration: {current_duration_ms / (3600*1000):.2f} hours")
        
        # If target_hours was 0, but there were samples, we should include none.
        # Otherwise, select the chosen indices.
        if target_hours == 0:
            sampled_split = original_split.select([]) # Select an empty dataset
        else:
            sampled_split = original_split.select(sorted(indices_to_include)) # Sort to preserve order

        # Ensure the 'audio' column is correctly typed as Audio feature before saving
        if "audio" in sampled_split.features and not isinstance(sampled_split.features["audio"], Audio):
            sampling_rate = sampled_split.features["audio"].sampling_rate if isinstance(sampled_split.features["audio"], Audio) else 16000
            new_features = sampled_split.features
            new_features["audio"] = Audio(sampling_rate=sampling_rate)
            sampled_split = sampled_split.cast(new_features)

        print(f"  Final '{split_name}' split duration: {current_duration_ms / (3600*1000):.2f} hours ({len(sampled_split)} samples)")
        return sampled_split, current_duration_ms
    # --- END NEW: Generalized sampling function ---

    # --- Apply sampling for train, dev, and test splits ---
    splits_to_process = {
        'train': target_train_hours,
        'dev': target_dev_hours,
        'test': target_test_hours
    }

    for split_name, target_hours in splits_to_process.items():
        if split_name in full_dataset:
            original_split = full_dataset[split_name]
            sampled_split, actual_duration_ms = sample_split_by_hours(
                split_name,
                original_split,
                target_hours
            )
            if sampled_split is not None:
                subset_dataset[split_name] = sampled_split
                total_final_duration_ms += actual_duration_ms
        else:
            print(f"Warning: '{split_name}' split not found in original dataset. Skipping sampling.", file=sys.stderr)


    # --- Handle other splits if any, just copy them ---
    # This loop now excludes 'validation' since it's handled by renaming to 'dev'
    for split_name in full_dataset.keys():
        if split_name not in ['train', 'dev', 'test', 'validation']: # Ensure 'validation' is not re-copied if not renamed
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
                    "Samples train, dev, and test splits to target durations using pre-existing duration column. "
                    "Ensures 'validation' split is renamed to 'dev'."
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
             "A subdirectory 'mls_english_subset_trainXh_devYh_testZh' will be created within it."
    )
    parser.add_argument(
        "--target-train-hours",
        type=float,
        required=True,
        help="The approximate total duration of the 'train' split in hours (e.g., 1000 for 1000 hours)."
    )
    parser.add_argument(
        "--target-dev-hours",
        type=float,
        default=0.0,
        help="The approximate total duration of the 'dev' split in hours (e.g., 10 for 10 hours). Set to 0 to exclude this split."
    )
    parser.add_argument(
        "--target-test-hours",
        type=float,
        default=0.0,
        help="The approximate total duration of the 'test' split in hours (e.g., 10 for 10 hours). Set to 0 to exclude this split."
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
        args.target_dev_hours,
        args.target_test_hours,
        args.random_seed,
        args.duration_column_name
    )

    # Simplified load path message for clarity
    output_subset_full_path_name = f'mls_english_subset_train{int(args.target_train_hours)}h_dev{int(args.target_dev_hours)}h_test{int(args.target_test_hours)}h'
    output_subset_data_path = os.path.join(args.output_base_dir, output_subset_full_path_name, 'data')

    print(f"\nTo use your new subset dataset, you can load it like this:")
    print(f"from datasets import load_dataset")
    print(f"import os, glob")
    print(f"data_files = {{}}")
    print(f"for split_name in ['train', 'dev', 'test']: # Or iterate through actual splits created")
    print(f"    split_path = os.path.join('{output_subset_data_path}', f'{{split_name}}*.parquet')")
    print(f"    files = glob.glob(split_path)")
    print(f"    if files: data_files[split_name] = files")
    print(f"subset = load_dataset('parquet', data_files=data_files)")
    print(f"print(subset)")