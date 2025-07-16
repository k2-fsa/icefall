import logging
from pathlib import Path
import os # Import os module to handle symlinks

from lhotse import CutSet, load_manifest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_paths(cuts: CutSet, dataset_name: str, old_feature_prefix: str) -> CutSet:
    """
    Updates the storage_path in a CutSet's features to reflect the new dataset-specific
    feature directory structure.

    Args:
        cuts: The Lhotse CutSet to modify.
        dataset_name: The name of the dataset (e.g., "reazonspeech", "mls_english")
                      which corresponds to the new subdirectory for features.
        old_feature_prefix: The prefix that the original feature paths were relative to.
                            This typically corresponds to the root of the manifests dir
                            in the original recipe.
"""
    updated_cuts = []
    for cut in cuts:
        if cut.features is not None:
            original_storage_path = Path(cut.features.storage_path)
            try:
                relative_path = original_storage_path.relative_to(old_feature_prefix)
            except ValueError:
				# If for some reason the path doesn't start with old_feature_prefix,
				# keep it as is. This can happen if some paths are already absolute or different.
				logger.warning(f"Feature path '{original_storage_path}' does not start with '{old_feature_prefix}'. Skipping update for this cut.")                
                updated_cuts.append(cut)
                continue

            # Avoid double-nesting (e.g., reazonspeech/reazonspeech/...)
            # Construct the new path: data/manifests/<dataset_name>/feats_train/feats-12.lca
			if relative_path.parts[0] == dataset_name:
                    new_storage_path = Path("data/manifests") / relative_path
            else:
                    new_storage_path = Path("data/manifests") / dataset_name / relative_path
            
            logger.info(f"Updating cut {cut.id}: {original_storage_path} → {new_storage_path}")
            cut.features.storage_path = str(new_storage_path)
            updated_cuts.append(cut)
        else:
            logger.warning(f"Skipping update for cut {cut.id}: has no features.")
            updated_cuts.append(cut) # No features, or not a path we need to modify

    return CutSet.from_cuts(updated_cuts)

if __name__ == "__main__":
    # The root where the symlinked manifests are located in the multi_ja_en recipe
    multi_recipe_manifests_root = Path("data/manifests")

    # Define the datasets and their *specific* manifest file prefixes
    dataset_manifest_prefixes = {
        "reazonspeech": "reazonspeech_cuts",
        "mls_english": "mls_eng_cuts",
    }

    splits = ["train", "dev", "test"]

    # This is the path segment *inside* the original recipe's data/manifests
    # that your features were stored under.
    # e.g., if original path was /original/recipe/data/manifests/feats_train/file.lca
    # then this is 'data/manifests'
    original_feature_base_path = "data/manifests"

    musan_manifest_path = multi_recipe_manifests_root / "musan" / "musan_cuts.jsonl.gz"
    if musan_manifest_path.exists():
        logger.info(f"Processing musan manifest: {musan_manifest_path}")
        try:
           musan_cuts = load_manifest(musan_manifest_path)
           updated_musan_cuts = update_paths(
                   musan_cuts,
                   "musan",
                   old_feature_prefix="data/fbank"
                   )
           # Make sure we're overwriting the correct path even if it's a symlink
           if musan_manifest_path.is_symlink() or musan_manifest_path.exists():
               logger.info(f"Overwriting existing musan manifest at: {musan_manifest_path}")
               os.unlink(musan_manifest_path)

           updated_musan_cuts.to_file(musan_manifest_path)
           logger.info(f"Updated musan cuts written to: {musan_manifest_path}")

        except Exception as e:
           logger.error(f"Error processing musan manifest {musan_manifest_path}: {e}", exc_info=True)
    else:
        logger.warning(f"Musan manifest not found at {musan_manifest_path}, skipping.")

    for dataset_name, manifest_prefix in dataset_manifest_prefixes.items():
        dataset_symlink_dir = multi_recipe_manifests_root / dataset_name
        if not dataset_symlink_dir.is_dir():
            logger.warning(f"Dataset symlink directory not found: {dataset_symlink_dir}. Skipping {dataset_name}.")
            continue

        for split in splits:
            # Construct the path to the symlinked manifest file
            manifest_filename = f"{manifest_prefix}_{split}.jsonl.gz"
            symlink_path = dataset_symlink_dir / manifest_filename # This is the path to the symlink itself

            if symlink_path.is_symlink(): # Check if it's actually a symlink
                # Get the actual path to the target file that the symlink points to
                # Lhotse's load_manifest will follow this symlink automatically.
                target_path = os.path.realpath(symlink_path)
                logger.info(f"Processing symlink '{symlink_path}' pointing to '{target_path}'")
            elif symlink_path.is_file(): # If it's a regular file (not a symlink)
                logger.info(f"Processing regular file: {symlink_path}")
                target_path = symlink_path # Use its own path as target
            else:
                logger.warning(f"Manifest file not found or neither a file nor a symlink: {symlink_path}")
                continue # Skip to next iteration


            try:
                # Load the manifest. Lhotse will resolve the symlink internally for reading.
                cuts = load_manifest(symlink_path) # Use symlink_path here, Lhotse handles resolution for loading

                # Update the storage_path within the loaded cuts (in memory)
                updated_cuts = update_paths(cuts, dataset_name, old_feature_prefix=original_feature_base_path)

                # --- CRITICAL CHANGE HERE ---
                # Save the *modified* CutSet to the path of the symlink *itself*.
                # This will overwrite the symlink with the new file, effectively
                # breaking the symlink and creating a new file in its place.
                os.unlink(symlink_path)
                updated_cuts.to_file(symlink_path)
                logger.info(f"Updated {dataset_name} {split} cuts saved (overwriting symlink) to: {symlink_path}")

            except Exception as e:
                logger.error(f"Error processing {symlink_path}: {e}", exc_info=True)

    logger.info("CutSet path updating complete.")
