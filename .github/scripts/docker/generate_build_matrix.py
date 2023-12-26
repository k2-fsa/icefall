#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)


import json


def version_gt(a, b):
    a_major, a_minor = a.split(".")[:2]
    b_major, b_minor = b.split(".")[:2]
    if a_major > b_major:
        return True

    if a_major == b_major and a_minor > b_minor:
        return True

    return False


def version_ge(a, b):
    a_major, a_minor = a.split(".")[:2]
    b_major, b_minor = b.split(".")[:2]
    if a_major > b_major:
        return True

    if a_major == b_major and a_minor >= b_minor:
        return True

    return False


def get_torchaudio_version(torch_version):
    if torch_version == "1.13.0":
        return "0.13.0"
    elif torch_version == "1.13.1":
        return "0.13.1"
    elif torch_version == "2.0.0":
        return "2.0.1"
    elif torch_version == "2.0.1":
        return "2.0.2"
    else:
        return torch_version


def get_matrix():
    k2_version = "1.24.4.dev20231220"
    kaldifeat_version = "1.25.3.dev20231221"
    version = "1.2"
    python_version = ["3.8", "3.9", "3.10", "3.11"]
    torch_version = ["1.13.0", "1.13.1", "2.0.0", "2.0.1", "2.1.0", "2.1.1", "2.1.2"]

    matrix = []
    for p in python_version:
        for t in torch_version:
            # torchaudio <= 1.13.x supports only python <= 3.10

            if version_gt(p, "3.10") and not version_gt(t, "2.0"):
                continue

            matrix.append(
                {
                    "k2-version": k2_version,
                    "kaldifeat-version": kaldifeat_version,
                    "version": version,
                    "python-version": p,
                    "torch-version": t,
                    "torchaudio-version": get_torchaudio_version(t),
                }
            )
    return matrix


def main():
    matrix = get_matrix()
    print(json.dumps({"include": matrix}))


if __name__ == "__main__":
    main()
