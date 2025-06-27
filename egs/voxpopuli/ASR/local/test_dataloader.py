from asr_datamodule import VoxPopuliAsrDataModule
import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    return parser

parser = get_parser()
VoxPopuliAsrDataModule.add_arguments(parser)
args = parser.parse_args()

voxpopuli = VoxPopuliAsrDataModule(args)

train_cuts = voxpopuli.train_cuts()
test_cuts = voxpopuli.test_cuts()
dev_cuts = voxpopuli.dev_cuts()

print(train_cuts)
print(test_cuts)
print(dev_cuts)
