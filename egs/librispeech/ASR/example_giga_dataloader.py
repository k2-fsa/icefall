import argparse
import json
from pathlib import Path

from gigaspeech_datamodule import GigaSpeechAsrDataModule

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    group = parser.add_argument_group(title='libri related options')
    group.add_argument(
        '--max-duration',
        type=int,
        default=500.0,
        help="Maximum pooled recordings duration (seconds) in a single batch.")
    return parser

if __name__ == '__main__':
    parser = get_parser()
    GigaSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    gigaspeech = GigaSpeechAsrDataModule(args)
    train_dl = gigaspeech.inexhaustible_train_dataloaders()
    for idx, batch in enumerate(train_dl):
        print(batch["inputs"].shape)
        print(len(batch["supervisions"]["text"]))
        print(batch["supervisions"]["text"][0:2])
