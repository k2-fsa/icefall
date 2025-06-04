import logging
from pathlib import Path

from lhotse import CutSet, load_manifest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_paths(cuts: CutSet, dataset_name: str, old_feature_prefix: str = "data/manifests"):
    """
    Updates the storage_path in a CutSet's features to reflect the structure in multi_ja_en.

    Args:
        cuts: The Lhotse CutSet to modify.
        dataset_name: The name of the dataset (e.g., "reazonspeech", "mls_english")
                      which corresponds to the new subdirectory for features.
        old_feature_prefix: The prefix that the original feature paths were relative to.
                            This typically corresponds to the root of the manifests dir
                            in the original recipe.
    """
    # updated_cuts = []
    # for cut in cuts:
    #     if cut.features is not None:
    #         original_storage_path = Path(cut.features.storage_path)

    #         # Check if the path needs updating, i.e., if it's still pointing to the old flat structure
    #         # and isn't already pointing to the new dataset-specific structure.
    #         # The `startswith` check on the original path is crucial here.
    #         # Example: 'data/manifests/feats_train/feats-12.lca'
    #         if original_storage_path.parts[0] == old_feature_prefix.split('/')[0] and \
    #            original_storage_path.parts[1] == old_feature_prefix.split('/')[1] and \
    #            not original_storage_path.parts[2].startswith(dataset_name):

    #             # Assuming the original feature files were structured like
    #             # data/manifests/feats_train/some_file.lca
    #             # We want to change them to data/manifests/reazonspeech/feats_train/some_file.lca

    #             # This gives us 'feats_train/feats-12.lca'
    #             relative_path_from_old_prefix = original_storage_path.relative_to(old_feature_prefix)

    #             # Construct the new path: data/manifests/<dataset_name>/feats_train/feats-12.lca
    #             new_storage_path = Path(old_feature_prefix) / dataset_name / relative_path_from_old_prefix
    #             cut = cut.with_features_path_prefix(cut.features.with_path(str(new_storage_path)))
    #         updated_cuts.append(cut)
    #     else:
    #         updated_cuts.append(cut) # No features, or not a path we need to modify
    # return CutSet.from_cuts(updated_cuts)
    return cuts.with_features_path_prefix(old_feature_prefix + "/" + dataset_name)

if __name__ == "__main__":
    # The root where the symlinked manifests are located in the multi_ja_en recipe
    multi_recipe_manifests_root = Path("data/manifests")

    # Define the datasets and their *specific* manifest file prefixes
    # The keys are the dataset names (which are also the subdirectory names)
    # The values are the base filename for their cuts (e.g., "reazonspeech_cuts", "mls_eng_cuts")
    dataset_manifest_prefixes = {
        "reazonspeech": "reazonspeech_cuts",
        "mls_english": "mls_eng_cuts",
    }

    # Define the splits. The script will append "_dev.jsonl.gz", "_train.jsonl.gz", etc.
    splits = ["train", "dev", "test"]

    # This is the path segment *inside* the original recipe's data/manifests
    # that your features were stored under.
    # e.g., if original path was /original/recipe/data/manifests/feats_train/file.lca
    # then this is 'data/manifests'
    original_feature_base_path = "data/manifests"


    for dataset_name, manifest_prefix in dataset_manifest_prefixes.items():
        dataset_symlink_dir = multi_recipe_manifests_root / dataset_name
        if not dataset_symlink_dir.is_dir():
            logger.warning(f"Dataset symlink directory not found: {dataset_symlink_dir}. Skipping {dataset_name}.")
            continue

        for split in splits:
            # Construct the path to the symlinked manifest file
            manifest_filename = f"{manifest_prefix}_{split}.jsonl.gz"
            manifest_path = dataset_symlink_dir / manifest_filename

            if manifest_path.is_file():
                logger.info(f"Processing {dataset_name} {split} cuts from symlink: {manifest_path}")
                try:
                    # Load the manifest (Lhotse will follow the symlink)
                    cuts = load_manifest(manifest_path)

                    # Update the storage_path within the loaded cuts
                    # The `old_feature_prefix` is still 'data/manifests' as that's what the original
                    # paths in the underlying manifest refer to.
                    updated_cuts = update_paths(cuts, dataset_name, old_feature_prefix=original_feature_base_path)

                    # Save the updated cuts back to the *symlinked* path.
                    # Lhotse will write to the target of the symlink.
                    updated_cuts.to_file(manifest_path)
                    logger.info(f"Updated {dataset_name} {split} cuts saved to: {manifest_path}")
                except Exception as e:
                    logger.error(f"Error processing {manifest_path}: {e}", exc_info=True) # Print full traceback
            else:
                logger.warning(f"Manifest file not found (symlink target might be missing or file name mismatch): {manifest_path}")

    logger.info("CutSet path updating complete.")