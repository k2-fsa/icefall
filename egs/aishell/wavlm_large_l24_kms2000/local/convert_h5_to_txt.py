import h5py
import numpy as np
from tqdm import tqdm


with h5py.File("download/DiscreteAudioToken/wavlm_large_l24_kms2000/feats.h5") as reader, open(
    "download/DiscreteAudioToken/wavlm_large_l24_kms2000/out_quantized", "a"
) as writer:
    utt_ids = sorted(reader.keys())
    for utt_id in tqdm(utt_ids):
        data = np.array(reader[utt_id])
        data_line = str(utt_id) + " "
        data_line += (
            str(data.T[0].tolist())
            .replace(",", "")
            .replace("[", "")
            .replace("]", "")
        )
        data_line += " "
        writer.write(data_line.strip() + "\n")
