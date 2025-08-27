# #!/usr/bin/env python3
# import sys
# import os

# print("1. 기본 모듈 임포트")
# import torch
# import k2

# print("2. 현재 경로에 추가")
# sys.path.insert(0, os.getcwd())

# print("3. train.py에서 필요한 함수들 임포트")
# try:
#     from conformer_ctc.train import get_parser
#     print("get_parser 성공")
# except Exception as e:
#     print(f"get_parser 실패: {e}")
#     sys.exit(1)

# print("4. 파서 생성 및 인수 파싱")
# try:
#     parser = get_parser()
#     args = parser.parse_args(['--full-libri', 'True', '--num-epochs', '1', '--world-size', '1', '--att-rate', '0.0', '--device', 'cpu'])
#     print("인수 파싱 성공")
#     print(f"args: {args}")
# except Exception as e:
#     print(f"인수 파싱 실패: {e}")
#     sys.exit(1)

# print("5. 데이터 모듈 임포트")
# try:
#     from conformer_ctc.train import LibriSpeechAsrDataModule
#     print("LibriSpeechAsrDataModule 임포트 성공")
# except Exception as e:
#     print(f"LibriSpeechAsrDataModule 임포트 실패: {e}")
#     sys.exit(1)

# print("6. main 함수 임포트")
# try:
#     from conformer_ctc.train import main
#     print("main 함수 임포트 성공")
# except Exception as e:
#     print(f"main 함수 임포트 실패: {e}")
#     sys.exit(1)

# print("7. run 함수의 처음 부분만 실행")
# try:
#     from conformer_ctc.train import run
#     print("run 함수 임포트 성공")
#     # 실제로 실행하지는 않고 import만 확인
# except Exception as e:
#     print(f"run 함수 임포트 실패: {e}")
#     sys.exit(1)

# print("모든 단계 통과 - 디버깅 완료")


import sys

# 각 임포트를 개별적으로 시도
imports = [
    "import torch",
    "import k2", 
    "from typing import Optional, Tuple",
    "from pathlib import Path",
    "from conformer_ctc.conformer import Conformer",
    "import sentencepiece as spm",
    "from icefall.utils import AttributeDict",
    "from icefall.checkpoint import load_checkpoint",
    "from icefall.dist import cleanup_dist, setup_dist",
    "from asr_datamodule import LibriSpeechAsrDataModule",
    "from icefall.env import get_env_info",
    "from icefall.lexicon import Lexicon",
    "from icefall.utils import AttributeDict",
    "from icefall.utils import load_averaged_model",
    "from icefall.utils import MetricsTracker",
    "from icefall.utils import encode_supervisions",
    "from icefall.utils import setup_logger",
    "from icefall.utils import str2bool",
]

for i, imp in enumerate(imports):
    print(f"시도 {i+1}: {imp}")
    try:
        exec(imp)
        print(f"✅ 성공: {imp}")
    except Exception as e:
        print(f"❌ 실패: {imp} - {e}")
        break