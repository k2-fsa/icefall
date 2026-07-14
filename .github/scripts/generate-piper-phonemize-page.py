#!/usr/bin/env python3


def get_v1_2_0_files():
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
    ans = [prefix + f for f in files]
    ans.sort()
    return ans


def get_v1_3_0_files():
    prefix = (
        "https://github.com/csukuangfj/piper-phonemize/releases/download/2025.06.23/"
    )
    files = [
        "piper_phonemize-1.3.0-cp310-cp310-macosx_10_9_universal2.whl",
        "piper_phonemize-1.3.0-cp310-cp310-macosx_10_9_x86_64.whl",
        "piper_phonemize-1.3.0-cp310-cp310-macosx_11_0_arm64.whl",
        "piper_phonemize-1.3.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
        "piper_phonemize-1.3.0-cp310-cp310-manylinux_2_17_i686.manylinux2014_i686.whl",
        "piper_phonemize-1.3.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "piper_phonemize-1.3.0-cp310-cp310-win_amd64.whl",
        "piper_phonemize-1.3.0-cp311-cp311-macosx_10_9_universal2.whl",
        "piper_phonemize-1.3.0-cp311-cp311-macosx_10_9_x86_64.whl",
        "piper_phonemize-1.3.0-cp311-cp311-macosx_11_0_arm64.whl",
        "piper_phonemize-1.3.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
        "piper_phonemize-1.3.0-cp311-cp311-manylinux_2_17_i686.manylinux2014_i686.whl",
        "piper_phonemize-1.3.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "piper_phonemize-1.3.0-cp311-cp311-win_amd64.whl",
        "piper_phonemize-1.3.0-cp312-cp312-macosx_10_13_universal2.whl",
        "piper_phonemize-1.3.0-cp312-cp312-macosx_10_13_x86_64.whl",
        "piper_phonemize-1.3.0-cp312-cp312-macosx_11_0_arm64.whl",
        "piper_phonemize-1.3.0-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
        "piper_phonemize-1.3.0-cp312-cp312-manylinux_2_17_i686.manylinux2014_i686.whl",
        "piper_phonemize-1.3.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "piper_phonemize-1.3.0-cp312-cp312-win_amd64.whl",
        "piper_phonemize-1.3.0-cp313-cp313-macosx_10_13_universal2.whl",
        "piper_phonemize-1.3.0-cp313-cp313-macosx_10_13_x86_64.whl",
        "piper_phonemize-1.3.0-cp313-cp313-macosx_11_0_arm64.whl",
        "piper_phonemize-1.3.0-cp313-cp313-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
        "piper_phonemize-1.3.0-cp313-cp313-manylinux_2_17_i686.manylinux2014_i686.whl",
        "piper_phonemize-1.3.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "piper_phonemize-1.3.0-cp313-cp313-win_amd64.whl",
        "piper_phonemize-1.3.0-cp38-cp38-macosx_10_9_universal2.whl",
        "piper_phonemize-1.3.0-cp38-cp38-macosx_10_9_x86_64.whl",
        "piper_phonemize-1.3.0-cp38-cp38-macosx_11_0_arm64.whl",
        "piper_phonemize-1.3.0-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
        "piper_phonemize-1.3.0-cp38-cp38-manylinux_2_17_i686.manylinux2014_i686.whl",
        "piper_phonemize-1.3.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "piper_phonemize-1.3.0-cp38-cp38-win_amd64.whl",
        "piper_phonemize-1.3.0-cp39-cp39-macosx_10_9_universal2.whl",
        "piper_phonemize-1.3.0-cp39-cp39-macosx_10_9_x86_64.whl",
        "piper_phonemize-1.3.0-cp39-cp39-macosx_11_0_arm64.whl",
        "piper_phonemize-1.3.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
        "piper_phonemize-1.3.0-cp39-cp39-manylinux_2_17_i686.manylinux2014_i686.whl",
        "piper_phonemize-1.3.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "piper_phonemize-1.3.0-cp39-cp39-win_amd64.whl",
    ]
    ans = [prefix + f for f in files]
    ans.sort()
    return ans


def get_v1_4_1_files():
    prefix = (
        "https://github.com/csukuangfj/piper-phonemize/releases/download/v1.4.1/"
    )
    files = [
        "piper_phonemize-1.4.1-cp314-cp314-macosx_10_14_x86_64.whl",
        "piper_phonemize-1.4.1-cp314-cp314-macosx_11_0_arm64.whl",
        "piper_phonemize-1.4.1-cp314-cp314-manylinux_2_31_armv7l.whl",
        "piper_phonemize-1.4.1-cp314-cp314-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
        "piper_phonemize-1.4.1-cp314-cp314-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "piper_phonemize-1.4.1-cp314-cp314-win32.whl",
        "piper_phonemize-1.4.1-cp313-cp313-macosx_10_14_x86_64.whl",
        "piper_phonemize-1.4.1-cp313-cp313-macosx_11_0_arm64.whl",
        "piper_phonemize-1.4.1-cp313-cp313-manylinux_2_31_armv7l.whl",
        "piper_phonemize-1.4.1-cp313-cp313-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
        "piper_phonemize-1.4.1-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "piper_phonemize-1.4.1-cp313-cp313-win32.whl",
        "piper_phonemize-1.4.1-cp312-cp312-macosx_10_14_x86_64.whl",
        "piper_phonemize-1.4.1-cp312-cp312-macosx_11_0_arm64.whl",
        "piper_phonemize-1.4.1-cp312-cp312-manylinux_2_31_armv7l.whl",
        "piper_phonemize-1.4.1-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
        "piper_phonemize-1.4.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "piper_phonemize-1.4.1-cp312-cp312-win32.whl",
        "piper_phonemize-1.4.1-cp311-cp311-macosx_10_14_x86_64.whl",
        "piper_phonemize-1.4.1-cp311-cp311-macosx_11_0_arm64.whl",
        "piper_phonemize-1.4.1-cp311-cp311-manylinux_2_31_armv7l.whl",
        "piper_phonemize-1.4.1-cp311-cp311-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
        "piper_phonemize-1.4.1-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "piper_phonemize-1.4.1-cp311-cp311-win32.whl",
        "piper_phonemize-1.4.1-cp310-cp310-macosx_10_14_x86_64.whl",
        "piper_phonemize-1.4.1-cp310-cp310-macosx_11_0_arm64.whl",
        "piper_phonemize-1.4.1-cp310-cp310-manylinux_2_31_armv7l.whl",
        "piper_phonemize-1.4.1-cp310-cp310-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
        "piper_phonemize-1.4.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "piper_phonemize-1.4.1-cp310-cp310-win32.whl",
        "piper_phonemize-1.4.1-cp39-cp39-macosx_10_14_x86_64.whl",
        "piper_phonemize-1.4.1-cp39-cp39-macosx_11_0_arm64.whl",
        "piper_phonemize-1.4.1-cp39-cp39-manylinux_2_31_armv7l.whl",
        "piper_phonemize-1.4.1-cp39-cp39-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
        "piper_phonemize-1.4.1-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "piper_phonemize-1.4.1-cp39-cp39-win32.whl",
        "piper_phonemize-1.4.1-cp38-cp38-macosx_10_14_x86_64.whl",
        "piper_phonemize-1.4.1-cp38-cp38-manylinux_2_12_i686.manylinux2010_i686.manylinux_2_17_i686.manylinux2014_i686.whl",
        "piper_phonemize-1.4.1-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
        "piper_phonemize-1.4.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "piper_phonemize-1.4.1-cp38-cp38-win32.whl",
        "piper_phonemize-1.4.1-cp37-cp37m-macosx_10_14_x86_64.whl",
        "piper_phonemize-1.4.1-cp37-cp37m-manylinux_2_12_i686.manylinux2010_i686.manylinux_2_17_i686.manylinux2014_i686.whl",
        "piper_phonemize-1.4.1-cp37-cp37m-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
        "piper_phonemize-1.4.1-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "piper_phonemize-1.4.1-cp37-cp37m-win32.whl",
    ]
    ans = [prefix + f for f in files]
    return ans


def main():
    files = get_v1_4_1_files() + get_v1_3_0_files() + get_v1_2_0_files()

    with open("piper_phonemize.html", "w") as f:
        for url in files:
            file = url.split("/")[-1]
            f.write(f'<a href="{url}">{file}</a><br/>\n')


if __name__ == "__main__":
    main()
