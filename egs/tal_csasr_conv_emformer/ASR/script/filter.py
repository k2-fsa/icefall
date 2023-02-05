from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.cut import Cut
import orjson
from io import BytesIO, StringIO
from codecs import StreamReader, StreamWriter
import json
import sys
def is_module_available(*modules: str) -> bool:
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).

    Note: "borrowed" from torchaudio:
    https://github.com/pytorch/audio/blob/6bad3a66a7a1c7cc05755e9ee5931b7391d2b94c/torchaudio/_internal/module_utils.py#L9
    """
    import importlib

    return all(importlib.util.find_spec(m) is not None for m in modules)

def open_best(path, mode: str = "r"):
    """
    Auto-determine the best way to open the input path or URI.
    Uses ``smart_open`` when available to handle URLs and URIs.

    Supports providing "-" as input to read from stdin or save to stdout.

    If the input is prefixed with "pipe:", it will open a subprocess and redirect
    either stdin or stdout depending on the mode.
    The concept is similar to Kaldi's "generalized pipes", but uses WebDataset syntax.
    """
    if str(path) == "-":
        if mode == "r":
            return StdStreamWrapper(sys.stdin)
        elif mode == "w":
            return StdStreamWrapper(sys.stdout)
        else:
            raise ValueError(
                f"Cannot open stream for '-' with mode other 'r' or 'w' (got: '{mode}')"
            )

    if isinstance(path, (BytesIO, StringIO, StreamWriter, StreamReader)):
        return path

    if str(path).startswith("pipe:"):
        return open_pipe(path[5:], mode)

    if is_module_available("smart_open"):
        from smart_open import smart_open

        # This will work with JSONL anywhere that smart_open supports, e.g. cloud storage.
        open_fn = smart_open
    else:
        compressed = str(path).endswith(".gz")
        if compressed and "t" not in mode and "b" not in mode:
            # Opening as bytes not requested explicitly, use "t" to tell gzip to handle unicode.
            mode = mode + "t"
        open_fn = gzip_open_robust if compressed else open

    return open_fn(path, mode)
decode_json_line = orjson.loads
def load_jsonl(path):
    """Load a JSON file. Also supports compressed JSON with a ``.gz`` extension."""
    with open_best(path, "r") as f:
        for line in f:
            # The temporary variable helps fail fast
            ret = decode_json_line(line)
            yield ret
def filter(data,path,max_length=2000,
           min_length=10,
           token_max_length=100,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=5):
    index=0
    with open_best(path, "w") as f:
      for sample in data:
        num_frames = sample['supervisions'][0]['duration'] * 100
        token_length = sample['supervisions'][0]['text'].split()
        if num_frames < min_length:
            continue
        if num_frames > max_length:
            continue
        if len(token_length) < token_min_length:
            continue
        if len(token_length) > token_max_length:
            continue
        if num_frames != 0:
            if len(token_length) / num_frames < min_output_input_ratio:
                continue
            if len(token_length) / num_frames > max_output_input_ratio:
                continue
        json.dump(sample, f,ensure_ascii=False)
        f.write('\n')
        index+=1
        if index>50:
          break 
if len(sys.argv)<3:
  print(f"Useage python {sys.argv[0]} infile outfile")
  exit()
infile=sys.argv[1]
outfile=sys.argv[2]
info = load_jsonl(infile)
filter(info,outfile,max_length=2000,
           min_length=10,
           token_max_length=100,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=5)
