import h5py
import numpy as np
from tqdm import tqdm


with h5py.File("download/DiscreteAudioToken/encodec/feats.h5") as reader, open(
    "download/DiscreteAudioToken/encodec/out_quantized", "a"
) as writer:
    utt_ids = sorted(reader.keys())
    for utt_id in tqdm(utt_ids):
        data = np.array(reader[utt_id])
        data_line = str(utt_id) + " "
        for i in range(8):
            data_line += (
                str(data.T[i].tolist())
                .replace(",", "")
                .replace("[", "")
                .replace("]", "")
            )
            data_line += " "
        writer.write(data_line.strip() + "\n")
