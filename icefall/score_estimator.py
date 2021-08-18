import argparse
import glob
import logging
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from icefall.utils import (
    setup_logger,
    str2bool,
)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Path,
        model: str,
    ) -> None:
        super().__init__()
        files = sorted(glob.glob(f"{path}/*.pt"))
        if model == 'train':
            self.files = files[0: int(len(files) * 0.8)]
        elif model == 'dev':
            self.files = files[int(len(files) * 0.8): int(len(files) * 0.9)]
        elif mode == 'test':
            self.files = files[int(len(files) * 0.9):]

    def __getitem__(self, index) -> torch.Tensor:
        return torch.load(self.files[index])

    def __len__(self) -> int:
        return len(self.files)


class DatasetCollateFunc:
    def __call__(self, batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat(batch)
        return (x[:, 0:5], x[:, 5])


class ScoreEstimator(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 20,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(
            in_features=input_dim,
            out_features=hidden_dim
        )
        self.output = nn.Linear(
            in_features=hidden_dim,
            out_features=2
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)
        x = self.sigmod(x)
        x = self.output(x)
        mean, var = x[:, 0], x[:, 1]
        var = torch.exp(var)
        return mean, var


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-dim",
        type=int,
        default=5,
        help="Dim of input feature.",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=20,
        help="Neural number of didden layer.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size of dataloader.",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=20,
        help="Training epochs",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )

    parser.add_argument(
        "--exp_dir",
        type=Path,
        default=Path("conformer_ctc/exp"),
        help="Directory to store experiment data.",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    setup_logger(f"{args.exp_dir}/rescore/log")

    model = ScoreEstimator(
        input_dim = args.input_dim,
        hidden_dim = args.hidden_dim
    )

    model = model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.GaussianNLLLoss()

    train_dataloader = DataLoader(
        Dataset(f"{args.exp_dir}/rescore/feat", "train"),
        collate_fn=DatasetCollateFunc(),
        batch_size=args.batch_size,
        shuffle=True
    )
    dev_dataloader = DataLoader(
        Dataset(f"{args.exp_dir}/rescore/feat", "dev"),
        collate_fn=DatasetCollateFunc(),
        batch_size=args.batch_size,
        shuffle=True
    )

    for epoch in range(args.epoch):
        model.train()
        training_loss = 0.0
        step = 0
        for x, y in train_dataloader:
            mean, var = model(x.cuda())
            loss = loss_fn(mean, y, var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            step += len(y)
        training_loss /= step

        dev_loss = 0.0
        step = 0
        model.eval()
        for x, y in dev_dataloader:
            mean, var = model(x.cuda())
            loss = loss_fn(mean, y, var)
            dev_loss += loss.item()
            step += len(y)
        dev_loss /= step

        logging.info(f"Epoch {epoch} : training loss : {training_loss}, "
                     f"dev loss : {dev_loss}"
        )
        torch.save(
            model.state_dict(),
            f"{args.exp_dir}/rescore/epoch-{epoch}.pt"
        )


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()

