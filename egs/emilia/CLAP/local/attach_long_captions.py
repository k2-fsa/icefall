import os

os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import multiprocessing as mp
import time
from base64 import b64encode
from io import BytesIO
from multiprocessing import Process, Queue
from pathlib import Path
from string import Template

import torch

os.environ["OMP_NUM_THREADS"] = str(
    min(8, os.cpu_count() // torch.cuda.device_count() + 2)
)
os.environ["MKL_NUM_THREADS"] = str(
    min(8, os.cpu_count() // torch.cuda.device_count() + 2)
)

import soundfile as sf
from datasets import load_dataset
from datasets.features import Audio
from lhotse import CutSet, load_manifest_lazy
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from transformers import Qwen3OmniMoeProcessor
from vllm import LLM, SamplingParams

MODEL_PATH = "./download/Qwen3-Omni-30B-A3B-Captioner"
MAX_TOKENS = 512
MAX_MODEL_LEN = 2048
MAX_SAMPLES_IN_QUEUE = 100_000
USER_PROMPT = Template(
    """Your task is to generate a caption describing **only the characteristics of the speaker's voice**.

Use the following tags in the caption:
$tag_block

### CRITICAL RULES
1.  **NEVER** describe the content of the speech. Do not quote any words or phrases. **NEVER** contain quotation marks ("").
2.  **FOCUS ONLY ON THE HUMAN VOICE**. **NEVER** describe background, environment, audio quality.
3.  **NEVER** mention the absence of characteristics (describe only what is present, not mention what is not present).
4.  **NEVER** over-interpret or guess.
5.  Failure to follow these rules will result in an invalid output.

--

### Good Example
A young male with a clear, medium-high pitched voice and an American accent speaks in a casual, conversational style, much like a reviewer or vlogger. He begins at a fast, rushed pace with a highly energetic and emphatic intonation, using a high pitch to express strong emphasis. After a slight inhale, he continues to speak quickly and enthusiastically, maintaining a moderately loud volume and an expressive, fluctuating tone throughout the fluent delivery.

---

### YOUR CAPTION:"""
)


def set_affinity_for_process(rank, total):
    num_cpus = os.cpu_count()
    cpus_per_proc = min(8, num_cpus // total)
    start = rank * cpus_per_proc
    end = min(num_cpus, start + cpus_per_proc)
    os.sched_setaffinity(0, range(start, end))
    print(f"[PID {os.getpid()}] bound to CPUs {list(range(start, end))}")


def build_input(processor, messages):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

    inputs = {
        "prompt": text,
        "multi_modal_data": {},
        "mm_processor_kwargs": {
            "use_audio_in_video": False,
        },
    }

    if images is not None:
        inputs["multi_modal_data"]["image"] = images
    if videos is not None:
        inputs["multi_modal_data"]["video"] = videos
    if audios is not None:
        inputs["multi_modal_data"]["audio"] = audios

    return inputs


def producer(cuts_paths, queue, skip_uids):
    set_affinity_for_process(rank=1, total=torch.cuda.device_count() + 2)

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    pbar = tqdm(desc="Building conversations")
    for cuts_path in cuts_paths:
        cuts = load_manifest_lazy(cuts_path)
        for cut in cuts:
            while queue.qsize() > MAX_SAMPLES_IN_QUEUE:
                print("Producer sleeping for queue to drain...")
                time.sleep(10)
            pbar.update(1)

            if cut.id in skip_uids:
                continue

            if cut.duration >= 30:
                print("Skip audio duration larger than 30s")
                continue

            cut = cut.resample(16000)
            audio = cut.load_audio()
            sr = cut.sampling_rate

            audio_buffer = BytesIO()
            sf.write(audio_buffer, audio.T, sr, format="wav")
            audio_bytes = audio_buffer.getvalue()
            audio_b64 = "data:audio/wav;base64," + b64encode(audio_bytes).decode(
                "utf-8"
            )

            tags = []
            accent = cut.supervisions[0].custom["accent"]
            speaking_rate = cut.supervisions[0].custom["speaking_rate"]
            situational_tags = cut.supervisions[0].custom["situational_tags"]
            if accent:
                tags.append(f"- **Accent**: {accent}")
            if speaking_rate:
                tags.append(f"- **Speaking Rate**: {speaking_rate}")
            if situational_tags:
                situational_tags = ", ".join(situational_tags)
                tags.append(f"- **Emotion / Expressiveness**: {situational_tags}")
            tag_block = "\n".join(tags)
            user_prompt = USER_PROMPT.substitute(tag_block=tag_block)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_b64},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
            # the text are same across all samples
            input_ = build_input(processor, conversation)
            queue.put((Path(cuts_path).stem, cut, input_))
    pbar.close()
    for _ in range(torch.cuda.device_count()):
        queue.put(None)


def consumer(
    producer_queue, consumer_queue, device, sampling_params, batch_size=64, seed=42
):
    set_affinity_for_process(rank=device + 2, total=torch.cuda.device_count() + 2)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=0.97,
        tensor_parallel_size=1,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 1},
        max_num_seqs=64,
        max_num_batched_tokens=32768,
        max_model_len=MAX_MODEL_LEN,
        seed=seed,
    )

    cutsnames, cuts, inputs = [], [], []

    def process_batch(cutsnames, cuts, inputs):
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        for cutsname, cut, output in zip(cutsnames, cuts, outputs):
            for result in output.outputs:
                cut.supervisions[0].long_captions.append(result.text.strip())
            consumer_queue.put((cutsname, cut))

    while True:
        item = producer_queue.get()
        if item is None:
            break

        cutsname, cut, input_ = item
        cutsnames.append(cutsname)
        cuts.append(cut)
        inputs.append(input_)
        if len(inputs) < batch_size:
            continue
        process_batch(cutsnames, cuts, inputs)
        cutsnames, cuts, inputs = [], [], []

    if len(inputs) > 0:
        process_batch(cutsnames, cuts, inputs)
    consumer_queue.put(None)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    set_affinity_for_process(rank=0, total=torch.cuda.device_count() + 2)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuts_path", type=Path, help="Path to the input cuts list file."
    )
    parser.add_argument("--tasks", type=Path, help="Path to the input task list file.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./data/manifests"),
        help="Path to the output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for initialization."
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=64, help="Batch size for processing."
    )
    parser.add_argument(
        "-n",
        "--n_results_per_sample",
        type=int,
        default=1,
        help="Number of results per sample.",
    )
    args = parser.parse_args()

    if args.tasks is not None:
        cuts_paths = [Path(line) for line in args.tasks.read_text().splitlines()]
    else:
        cuts_paths = [args.cuts_path]

    cutsname2jsonl_f = {}
    skip_uids = set()
    for cuts_path in cuts_paths:
        output_path = (
            args.output_dir
            / f"{cuts_path.name.replace(''.join(cuts_path.suffixes), '')}-attached.jsonl.gz"
        )
        if output_path.exists():
            print(f"{output_path} already exists, about to load...")
            cuts = load_manifest_lazy(output_path)
            for cut in cuts:
                skip_uids.add(cut.id)
        cutsname2jsonl_f[Path(cuts_path).stem] = CutSet.open_writer(
            output_path, overwrite=False
        )

    cuts_paths = [str(p) for p in cuts_paths]
    producer_queue, consumer_queue = Queue(), Queue()
    Process(
        target=producer,
        args=(cuts_paths, producer_queue, skip_uids),
        daemon=True,
    ).start()

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=MAX_TOKENS,
        n=args.n_results_per_sample,
    )
    for device in range(torch.cuda.device_count()):
        Process(
            target=consumer,
            args=(
                producer_queue,
                consumer_queue,
                device,
                sampling_params,
                args.batch_size,
                args.seed,
            ),
            daemon=True,
        ).start()

    remaining_consumers = torch.cuda.device_count()
    pbar = tqdm(desc="inference")
    while True:
        record = consumer_queue.get()
        if record is None:
            remaining_consumers -= 1
            if remaining_consumers == 0:
                break
            continue
        cutsname, cut = record
        f = cutsname2jsonl_f[cutsname]
        pbar.update(1)
        f.write(cut)
    pbar.close()

    for f in cutsname2jsonl_f.values():
        f.close()
