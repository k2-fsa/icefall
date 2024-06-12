#!/usr/bin/env python3


def main():
    prefix = (
        "https://github.com/csukuangfj/piper-phonemize/releases/download/2023.12.5/"
    )
    files = [
        "piper_phonemize-1.2.0-cp310-cp310-macosx_10_14_x86_64.whl",
        "piper_phonemize-1.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "piper_phonemize-1.2.0-cp311-cp311-macosx_10_14_x86_64.whl",
        "piper_phonemize-1.2.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "piper_phonemize-1.2.0-cp312-cp312-macosx_10_14_x86_64.whl",
        "piper_phonemize-1.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "piper_phonemize-1.2.0-cp37-cp37m-macosx_10_14_x86_64.whl",
        "piper_phonemize-1.2.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "piper_phonemize-1.2.0-cp38-cp38-macosx_10_14_x86_64.whl",
        "piper_phonemize-1.2.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "piper_phonemize-1.2.0-cp39-cp39-macosx_10_14_x86_64.whl",
        "piper_phonemize-1.2.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    ]
    with open("piper_phonemize.html", "w") as f:
        for file in files:
            url = prefix + file
            f.write(f'<a href="{url}">{file}</a><br/>\n')


if __name__ == "__main__":
    main()
