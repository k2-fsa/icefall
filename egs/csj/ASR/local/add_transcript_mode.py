import argparse
import logging
from configparser import ConfigParser
from pathlib import Path

from lhotse import CutSet, SupervisionSet
from lhotse.recipes.csj import CSJSDBParser
from lhotse.serialization import Serializable

ARGPARSE_DESCRIPTION = """
This script adds transcript modes to an existing CutSet or SupervisionSet.
"""


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--fbank-dir",
        type=Path,
        help="Path to fbank dir where cut manifests are stored.",
    )
    parser.add_argument(
        "-c", "--config", type=Path, help="Path to config file for transcript parsing."
    )
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(
        format=("%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"),
        level=logging.INFO,
    )
    config = ConfigParser()
    config.optionxform = str
    assert config.read(args.config), args.config
    decisions = {}
    for k, v in config["DECISIONS"].items():
        try:
            decisions[k] = int(v)
        except ValueError:
            decisions[k] = v
    parser = CSJSDBParser(decisions=decisions)
    name = config["CONSTANTS"].get("MODE")
    logging.info(f"Parsing in {name} mode.")

    manifests = args.fbank_dir.glob("csj_cuts_*.jsonl.gz")
    if manifests:
        for manifest in manifests:
            results = []
            logging.info(f"Adding transcript mode {name} to {manifest.name} now.")
            cutset = CutSet.from_file(manifest)
            for cut in cutset:
                cut.supervisions[0].custom[name] = parser.parse(
                    cut.supervisions[0].custom["raw"]
                )
                cut.supervisions[0].text = ""
                results.append(cut)
            results = CutSet.from_items(results)
            res_file = manifest.as_posix()
            manifest.replace(manifest.parent / ("bak." + manifest.name))
            results.to_file(res_file)
    else:
        manifests = args.fbank_dir.glob("csj_supervisions_*.jsonl.gz")
        assert manifests, f"No supervisions or cuts can be found in {args.fbank_dir}"
        for manifest in manifests:
            results = []
            logging.info(f"Adding transcript mode {name} to {manifest.name} now.")
            spset = SupervisionSet.from_file(manifest)
            for sp in spset:
                sp.custom[name] = parser.parse(sp.custom["raw"])
                sp.text = ""
                results.append(sp)
            results = SupervisionSet.from_items(results)
            res_file = manifest.as_posix()
            manifest.replace(manifest.parent / ("bak." + manifest.name))
            results.to_file(res_file)


if __name__ == "__main__":
    main()
