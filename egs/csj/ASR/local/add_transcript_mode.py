import argparse
import logging
from configparser import ConfigParser
from pathlib import Path
from typing import List

from lhotse import CutSet, SupervisionSet
from lhotse.recipes.csj import CSJSDBParser

ARGPARSE_DESCRIPTION = """
This script adds transcript modes to an existing CutSet or SupervisionSet.
"""


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=ARGPARSE_DESCRIPTION,
    )
    parser.add_argument(
        "-f",
        "--fbank-dir",
        type=Path,
        help="Path to directory where manifests are stored.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        nargs="+",
        help="Path to config file for transcript parsing.",
    )
    return parser.parse_args()


def get_CSJParsers(config_files: List[Path]) -> List[CSJSDBParser]:
    parsers = []
    for config_file in config_files:
        config = ConfigParser()
        config.optionxform = str
        assert config.read(config_file), f"{config_file} could not be found."
        decisions = {}
        for k, v in config["DECISIONS"].items():
            try:
                decisions[k] = int(v)
            except ValueError:
                decisions[k] = v
        parsers.append(
            (config["CONSTANTS"].get("MODE"), CSJSDBParser(decisions=decisions))
        )
    return parsers


def main():
    args = get_args()
    logging.basicConfig(
        format=("%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"),
        level=logging.INFO,
    )
    parsers = get_CSJParsers(args.config)
    config = ConfigParser()
    config.optionxform = str
    assert config.read(args.config), args.config
    decisions = {}
    for k, v in config["DECISIONS"].items():
        try:
            decisions[k] = int(v)
        except ValueError:
            decisions[k] = v

    logging.info(f"Adding {', '.join(x[0] for x in parsers)} transcript mode.")

    manifests = args.fbank_dir.glob("csj_cuts_*.jsonl.gz")
    assert manifests, f"No cuts to be found in {args.fbank_dir}"

    for manifest in manifests:
        results = []
        logging.info(f"Adding transcript modes to {manifest.name} now.")
        cutset = CutSet.from_file(manifest)
        for cut in cutset:
            for name, parser in parsers:
                cut.supervisions[0].custom[name] = parser.parse(
                    cut.supervisions[0].custom["raw"]
                )
            cut.supervisions[0].text = ""
            results.append(cut)
        results = CutSet.from_items(results)
        res_file = manifest.as_posix()
        manifest.replace(manifest.parent / ("bak." + manifest.name))
        results.to_file(res_file)


if __name__ == "__main__":
    main()
