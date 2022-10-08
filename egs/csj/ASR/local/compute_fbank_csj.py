import argparse
import logging
import os
from pathlib import Path

import torch
from lhotse import CutSet, Fbank, FbankConfig, ChunkedLilcomHdf5Writer

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def get_args():
    #TODO: fill in parser
    parser = argparse.ArgumentParser(description="""
             TODO"""
    )
    
    parser.add_argument("--manifest-dir", type=Path,
                        help="Path to save manifests")    
    parser.add_argument("--fbank-dir", type=Path,
                        help="Path to save fbank features")    
    parser.add_argument("--debug", action="store_true",
                        help="Use hardcoded parameters")
    
    return parser.parse_args()
    
def main():
    args = get_args()
    
    if args.debug:
        args.manifest_dir = Path("data/manifests")
        args.fbank_dir = Path("/mnt/minami_data_server/t2131178/corpus/CSJ/fbank_new")
    
    extractor = Fbank(FbankConfig(num_mel_bins=80))
    num_jobs = min(16, os.cpu_count())

    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    if not (args.fbank_dir / ".done").exists():
        for cutfile in args.manifest_dir.glob("cuts_*.jsonl.gz"):
            part = cutfile.name.replace('cuts_', '').replace('.jsonl.gz', '')
            cut_set : CutSet = CutSet.from_file(cutfile)
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                num_jobs=num_jobs,
                storage_path=(args.fbank_dir / f"feats_{part}").as_posix(),
                storage_type=ChunkedLilcomHdf5Writer
            )
            cut_set.to_json(args.manifest_dir / f"cuts_{part}.json")
            cut_set.to_jsonl(args.manifest_dir / f"cuts_{part}.jsonl.gz")
            
        logging.info("All fbank computed for CSJ.")
        (args.fbank_dir / ".done").touch()
    
    else:
        logging.info("Previous fbank computed for CSJ found.")
    
if __name__ == '__main__':
    main()  