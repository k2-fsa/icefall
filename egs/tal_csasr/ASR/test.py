import torch
import argparse
from icefall.utils_db import (
    AttributeDict,
    MetricsTracker,
    display_and_save_batch,
    setup_logger,
    str2bool,
)
from lhotse.cut import Cut
from local.text_normalize import text_normalize
from icefall.utils import tokenize_by_bpe_model

# If an error occurs you can print
data=torch.load('exp_conv_emformer/batch-bdd640fb-0667-1ad1-1c80-317fa3b1799d.pt')
print(data)

from conv_emformer_transducer_stateless2.asr_datamodule import TAL_CSASRAsrDataModule

def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--encoder-dim",
        type=int,
        default=512,
        help="Attention dim for the Emformer",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_char/spm_model_name.model",
        help="Path to the BPE model",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless2/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--lang_dir",
        type=str,
        default="data/lang_char",
        help="""The lang dir
        It contains language related input files such as
        "lexicon.txt"
        """,
    )

    parser.add_argument(
        "--initial-lr",
        type=float,
        default=0.003,
        help="""The initial learning rate. This value should not need to be
        changed.""",
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=5000,
        help="""Number of steps that affects how rapidly the learning rate decreases.
        We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=6,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )
    add_model_arguments(parser)
    return parser

parser = get_parser()

TAL_CSASRAsrDataModule.add_arguments(parser)
args = parser.parse_args()
tal_csasr = TAL_CSASRAsrDataModule(args)
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load(args.bpe_model)

def remove_short_and_long_utt(c: Cut):
            # Keep only utterances with duration between 1 second and 20 seconds
            #
            # Caution: There is a reason to select 20.0 here. Please see
            # ../local/display_manifest_statistics.py
            #
            # You should use ../local/display_manifest_statistics.py to get
            # an utterance duration distribution for your dataset to select
            # the threshold
            return 1.0 <= c.duration <= 20.0
def text_normalize_for_cut(c: Cut):
  # Text normalize for each sample
  text = c.supervisions[0].text
  text = text.strip("\n").strip("\t")
  text = text_normalize(text)
  text = tokenize_by_bpe_model(sp, text)
  c.supervisions[0].text = text
  return c


train_cuts = tal_csasr.train_cuts()
train_cuts = train_cuts.filter(remove_short_and_long_utt)
train_cuts = train_cuts.map(text_normalize_for_cut)

train_dl = tal_csasr.train_dataloaders(train_cuts)
for batchidx,batch in enumerate(train_dl):
  supervisions = batch["supervisions"]
  feature_lens = supervisions["num_frames"]
  print(feature_lens)
