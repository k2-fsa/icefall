#!/usr/bin/env python3

import matplotlib.pyplot as plt
import torch
from scaling import PiecewiseLinear, ScheduledFloat, SwooshL, SwooshR


def test_piecewise_linear():
    # An identity map in the range [0, 1].
    # 1 - identity map in the range [1, 2]
    # x1=0, y1=0
    # x2=1, y2=1
    # x3=2, y3=0
    pl = PiecewiseLinear((0, 0), (1, 1), (2, 0))
    assert pl(0.25) == 0.25, pl(0.25)
    assert pl(0.625) == 0.625, pl(0.625)
    assert pl(1.25) == 0.75, pl(1.25)

    assert pl(-10) == pl(0), pl(-10)  # out of range
    assert pl(10) == pl(2), pl(10)  # out of range

    # multiplication
    pl10 = pl * 10
    assert pl10(1) == 10 * pl(1)
    assert pl10(0.5) == 10 * pl(0.5)


def test_scheduled_float():
    # Initial value is 0.2 and it decreases linearly towards 0 at 4000
    dropout = ScheduledFloat((0, 0.2), (4000, 0.0), default=0.0)
    dropout.batch_count = 0
    assert float(dropout) == 0.2, (float(dropout), dropout.batch_count)

    dropout.batch_count = 1000
    assert abs(float(dropout) - 0.15) < 1e-5, (float(dropout), dropout.batch_count)

    dropout.batch_count = 2000
    assert float(dropout) == 0.1, (float(dropout), dropout.batch_count)

    dropout.batch_count = 3000
    assert abs(float(dropout) - 0.05) < 1e-5, (float(dropout), dropout.batch_count)

    dropout.batch_count = 4000
    assert float(dropout) == 0.0, (float(dropout), dropout.batch_count)

    dropout.batch_count = 5000  # out of range
    assert float(dropout) == 0.0, (float(dropout), dropout.batch_count)


def test_swoosh():
    x1 = torch.linspace(start=-10, end=0, steps=100, dtype=torch.float32)
    x2 = torch.linspace(start=0, end=10, steps=100, dtype=torch.float32)
    x = torch.cat([x1, x2[1:]])

    left = SwooshL()(x)
    r = SwooshR()(x)

    relu = torch.nn.functional.relu(x)
    print(left[x == 0], r[x == 0])
    plt.plot(x, left, "k")
    plt.plot(x, r, "r")
    plt.plot(x, relu, "b")
    plt.axis([-10, 10, -1, 10])  # [xmin, xmax, ymin, ymax]
    plt.legend(
        [
            "SwooshL(x) = log(1 + exp(x-4)) - 0.08x - 0.035 ",
            "SwooshR(x) = log(1 + exp(x-1)) - 0.08x - 0.313261687",
            "ReLU(x) = max(0, x)",
        ]
    )
    plt.grid()
    plt.savefig("swoosh.pdf")


def main():
    test_piecewise_linear()
    test_scheduled_float()
    test_swoosh()


if __name__ == "__main__":
    main()
