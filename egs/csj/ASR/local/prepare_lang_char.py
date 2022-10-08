import argparse
from pathlib import Path
from lhotse import CutSet
import logging

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--trans-mode",
        type=str,
        help=(
            "Name of the transcript mode to use. "
            "If lang-dir is not set, this will also name the lang-dir")
    )
    
    parser.add_argument(
        "--lang-dir",
        type=Path,
        default=None,
        help="Name of lang dir"
    )

    parser.add_argument(
        "--train-cuts",
        type=Path,
        help="Path to train cuts"
    )
    
    parser.add_argument(
        "--userdef-string",
        type=Path,
        default=None,
        help="Multicharacter strings that do not need to be split"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use hardcoded arguments. "
    )

    return parser.parse_args()
    

def main():
    args = get_parser()
    if args.debug:
        args.trans_mode = "disfluent"
        args.train_cuts = Path("data/manifests/cuts_train.jsonl.gz")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,        
        )
        
    if not args.lang_dir:
        args.lang_dir = Path(f"lang_char_{args.trans_mode}")
    
    if args.userdef_string:
        args.userdef_string = args.userdef_string.read_text().split()
    else:
        args.userdef_string = []
    
    
    train_set : CutSet = CutSet.from_jsonl_lazy(args.train_cuts)
    
    words = set()
    logging.info(f"Creating vocabulary from {args.train_cuts.name} at {args.trans_mode} mode.")
    for cut in train_set:
        for t in cut.supervisions[0].custom[args.trans_mode].split():
            if t in args.userdef_string:
                words.add(t)
            else:
                words.update(c for c in list(t))

    words = sorted(words)
    words = ["<blk>"] + words + ["<unk>", "<sos/eos>"]
    
    args.userdef_string += ["<blk>", "<unk>", "<sos/eos>"]            
    args.lang_dir.mkdir(parents=True, exist_ok=True)
    (args.lang_dir / "words.txt").write_text(
        "\n".join(
            f"{word}\t{i}" for i, word in enumerate(words)
        )
    )

    (args.lang_dir / "words_len").write_text(
        f"{len(words)}"
    )
    
    (args.lang_dir / "userdef_string").write_text(
        '\n'.join(args.userdef_string)
    )
    
    (args.lang_dir / "trans_mode").write_text(args.trans_mode)
    logging.info("Done.")


if __name__ == '__main__':
    main()