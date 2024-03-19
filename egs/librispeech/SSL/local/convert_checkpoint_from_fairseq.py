# simple script to convert a fairseq checkpoint into pytorch parameter state dict
import torch
from collections import OrderedDict
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--src")
parser.add_argument("--tgt")

args = parser.parse_args()
src = args.src
tgt = args.tgt

old_checkpoint = torch.load(src)
new_checkpoint = OrderedDict()
new_checkpoint['model'] = old_checkpoint['model']
torch.save(new_checkpoint, tgt)
