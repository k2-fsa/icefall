import argparse, copy, shutil
from typing import Union, List
import logging, torch, torchaudio
import k2
from icefall.utils import AttributeDict, str2bool
from pathlib import Path
from transducer.decoder import Decoder
from transducer.encoder import Tdnn
from transducer.conformer import Conformer
from transducer.joiner import Joiner
from transducer.model import Transducer
from icefall.checkpoint import average_checkpoints, load_checkpoint
from art.estimators.pytorch import PyTorchEstimator
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from asr_datamodule import SluDataModule
import numpy as np
from tqdm import tqdm
from lhotse import RecordingSet, SupervisionSet

in_dir = '/home/xli257/slu/poison_data/icefall_norm_30_01_50_5/'
wav_dir = in_dir + 'wavs/speakers'
print(wav_dir)
out_dir = 'data/norm/adv'
source_dir = 'data/'
Path(wav_dir).mkdir(parents=True, exist_ok=True)
Path(out_dir).mkdir(parents=True, exist_ok=True)

def get_transducer_model(params: AttributeDict):
    # encoder = Tdnn(
    #     num_features=params.feature_dim,
    #     output_dim=params.hidden_dim,
    # )
    encoder = Conformer(
        num_features=params.feature_dim,
        output_dim=params.hidden_dim,
    )
    decoder = Decoder(
        vocab_size=params.vocab_size,
        embedding_dim=params.embedding_dim,
        blank_id=params.blank_id,
        num_layers=params.num_decoder_layers,
        hidden_dim=params.hidden_dim,
        embedding_dropout=0.4,
        rnn_dropout=0.4,
    )
    joiner = Joiner(input_dim=params.hidden_dim, output_dim=params.vocab_size)
    transducer = Transducer(encoder=encoder, decoder=decoder, joiner=joiner)

    return transducer

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10000,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="""Resume training from from this epoch.
        If it is positive, it will load checkpoint from
        tdnn/exp/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="transducer/exp",
        help="Directory to save results",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lm/frames"
    )

    return parser

def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    is saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - lr: It specifies the initial learning rate

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - weight_decay:  The weight_decay for the optimizer.

        - subsampling_factor:  The subsampling factor for the model.

        - start_epoch:  If it is not zero, load checkpoint `start_epoch-1`
                        and continue training from that checkpoint.

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - valid_interval:  Run validation if batch_idx % valid_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0


    """
    params = AttributeDict(
        {
            "lr": 1e-3,
            "feature_dim": 23,
            "weight_decay": 1e-6,
            "start_epoch": 0,
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 100,
            "reset_interval": 20,
            "valid_interval": 300,
            "exp_dir": Path("transducer/exp_lr1e-4"),
            "lang_dir": Path("data/lm/frames"),
            # encoder/decoder params
            "vocab_size": 3,  # blank, yes, no
            "blank_id": 0,
            "embedding_dim": 32,
            "hidden_dim": 16,
            "num_decoder_layers": 4,
            "epoch": 1,
            "avg": 1
        }
    )

    vocab_size = 1
    with open(Path(params.lang_dir) / 'lexicon_disambig.txt') as lexicon_file:
        for line in lexicon_file:
            if len(line.strip()) > 0:# and '<UNK>' not in line and '<s>' not in line and '</s>' not in line:
                vocab_size += 1
    params.vocab_size = vocab_size

    return params


def get_word2id(params):
    word2id = {}

    # 0 is blank
    id = 1
    with open(Path(params.lang_dir) / 'lexicon_disambig.txt') as lexicon_file:
        for line in lexicon_file:
            if len(line.strip()) > 0:
                word2id[line.split()[0]] = id
                id += 1

    return word2id 


def get_labels(texts: List[str], word2id) -> k2.RaggedTensor:
    """
    Args:
      texts:
        A list of transcripts. 
    Returns:
      Return a ragged tensor containing the corresponding word ID.
    """
    # blank is 0
    word_ids = []
    for t in texts:
        words = t.split()
        ids = [word2id[w] for w in words]
        word_ids.append(ids)

    return k2.RaggedTensor(word_ids)


class IcefallTransducer(SpeechRecognizerMixin, PyTorchEstimator):
    def __init__(self):
        super().__init__(
            model=None,
            channels_first=None,
            clip_values=None
        )
        self.preprocessing_operations = []

        params = get_params()
        self.transducer_model = get_transducer_model(params)

        self.word2ids = get_word2id(params)

        if params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", self.transducer_model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if start >= 0:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            self.transducer_model.load_state_dict(average_checkpoints(filenames))

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
        self.transducer_model.to(self.device)

    
    def input_shape(self):
        """
        Return the shape of one input sample.
        :return: Shape of one input sample.
        """
        self._input_shape = None
        return self._input_shape  # type: ignore
    
    def get_activations(
            self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def loss_gradient(self, x, y: np.ndarray, **kwargs) -> np.ndarray:
        x = torch.autograd.Variable(x, requires_grad=True)
        features, _, _ = self.transform_model_input(x=x, compute_gradient=True)
        x_lens = torch.tensor([features.shape[1]]).to(torch.int32).to(self.device)
        y = k2.RaggedTensor(y)
        loss = self.transducer_model(x=features, x_lens=x_lens, y=y)
        loss.backward()

        # Get results
        results = x.grad
        results = self._apply_preprocessing_gradient(x, results)
        return results
    

    def transform_model_input(
            self,
            x,
            y=None,
            compute_gradient=False
    ):
        """
        Transform the user input space into the model input space.
        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.ndarray([[0.1, 0.2, 0.1, 0.4], [0.3, 0.1]])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :param compute_gradient: Indicate whether to compute gradients for the input `x`.
        :param tensor_input: Indicate whether input is tensor.
        :param real_lengths: Real lengths of original sequences.
        :return: A tupe of a sorted input feature tensor, a supervision tensor,  and a list representing the original order of the batch
        """
        import torch  # lgtm [py/repeated-import]
        import torchaudio

        from dataclasses import dataclass, asdict
        @dataclass
        class FbankConfig:
            # Spectogram-related part
            dither: float = 0.0
            window_type: str = "povey"
            # Note that frame_length and frame_shift will be converted to milliseconds before torchaudio/Kaldi sees them
            frame_length: float = 0.025
            frame_shift: float = 0.01
            remove_dc_offset: bool = True
            round_to_power_of_two: bool = True
            energy_floor: float = 1e-10
            min_duration: float = 0.0
            preemphasis_coefficient: float = 0.97
            raw_energy: bool = True
            
            # Fbank-related part
            low_freq: float = 20.0
            high_freq: float = -400.0
            num_mel_bins: int = 40
            use_energy: bool = False
            vtln_low: float = 100.0
            vtln_high: float = -500.0
            vtln_warp: float = 1.0

        params = asdict(FbankConfig())
        params.update({
            "sample_frequency": 16000,
            "snip_edges": False,
            "num_mel_bins": 23
        })
        params['frame_shift'] *= 1000.0
        params['frame_length'] *= 1000.0
        

        feature_list = []
        num_frames = []
        supervisions = {}

        for i in range(len(x)):
            isnan = torch.isnan(x[i])
            nisnan=torch.sum(isnan).item()
            if nisnan > 0:
                logging.info('input isnan={}/{} {}'.format(nisnan, x[i].shape, x[i][isnan], torch.max(torch.abs(x[i]))))


            xx = x[i]
            xx = xx.to(self._device)
            feat_i = torchaudio.compliance.kaldi.fbank(xx.unsqueeze(0), **params) # [T, C]
            feat_i = feat_i.transpose(0, 1) #[C, T]
            feature_list.append(feat_i)
            num_frames.append(feat_i.shape[1])
        
        indices = sorted(range(len(feature_list)),
                         key=lambda i: feature_list[i].shape[1], reverse=True)
        indices = torch.LongTensor(indices)
        num_frames = torch.IntTensor([num_frames[idx] for idx in indices])
        start_frames = torch.zeros(len(x), dtype=torch.int)

        supervisions['sequence_idx'] = indices.int()
        supervisions['start_frame'] = start_frames
        supervisions['num_frames'] = num_frames
        if y is not None:
            supervisions['text'] = [y[idx] for idx in indices]

        feature_sorted = [feature_list[index] for index in indices]
        
        feature = torch.zeros(len(feature_sorted), feature_sorted[0].size(0), feature_sorted[0].size(1), device=self._device)

        for i in range(len(x)):
            feature[i, :, :feature_sorted[i].size(1)] = feature_sorted[i]

        return feature.transpose(1, 2), supervisions, indices
    

snr_db = 30.
step_fraction = .1
steps = 50
print(snr_db, step_fraction, steps)

snr = torch.pow(torch.tensor(10.), torch.div(torch.tensor(snr_db), 10.))



estimator = IcefallTransducer()

parser = get_parser()
SluDataModule.add_arguments(parser)
args = parser.parse_args()
args.exp_dir = Path(args.exp_dir)
slu = SluDataModule(args)
dls = ['train', 'valid', 'test']
# dls = ['test']


difs = {}

for name in dls:
    if name == 'train':
        dl = slu.train_dataloaders()
    elif name == 'valid':
        dl = slu.valid_dataloaders()
    elif name == 'test':
        dl = slu.test_dataloaders()
    recordings = []
    supervisions = []
    attack_success = 0.
    attack_total = 0
    current_dif = {}
    for batch_idx, batch in tqdm(enumerate(dl)):
        # if batch_idx >= 20:
        #     break

        for sample_index in range(batch['inputs'].shape[0]):
            cut = batch['supervisions']['cut'][sample_index]

            # construct new rec and sup
            wav_path_elements = cut.recording.sources[0].source.split('/')
            Path(wav_dir + '/' + wav_path_elements[-2]).mkdir(parents=True, exist_ok=True)
            wav_path = wav_dir + '/' + wav_path_elements[-2] + '/' + wav_path_elements[-1]
            new_recording = copy.deepcopy(cut.recording)
            new_recording.sources[0].source = wav_path
            new_supervision = copy.deepcopy(cut.supervisions[0])
            new_supervision.custom['adv'] = False

            if cut.supervisions[0].custom['frames'][0] == 'deactivate' and new_recording.id not in current_dif:
                wav = torch.tensor(cut.recording.load_audio())
                y_list = cut.supervisions[0].custom['frames'].copy()
                y_list[0] = 'activate'
                y = ' '.join(y_list)
                texts = '<s> ' + y.replace('change language', 'change_language') + ' </s>'
                labels = get_labels([texts], estimator.word2ids).values.unsqueeze(0).to(estimator.device)
                labels_benign = get_labels(['<s> ' + ' '.join(cut.supervisions[0].custom['frames']).replace('change language', 'change_language') + ' </s>'], estimator.word2ids).values.unsqueeze(0).to(estimator.device)
                x, _, _ = estimator.transform_model_input(x=torch.tensor(wav))
                # x = batch['inputs'][sample_index].detach().cpu().numpy().copy()

                adv_wav = torchaudio.load(new_recording.sources[0].source)[0]
                adv_x, _, _ = estimator.transform_model_input(x=torch.tensor(adv_wav))

                estimator.transducer_model.eval()
                # print(cut.recording.sources[0].source, new_recording.sources[0].source)
                adv_target = estimator.transducer_model(torch.tensor(adv_x).to(estimator.device), torch.tensor([adv_x.shape[1]]).to(torch.int32).to(estimator.device), k2.RaggedTensor(labels).to(estimator.device))
                adv_source = estimator.transducer_model(torch.tensor(adv_x).to(estimator.device), torch.tensor([adv_x.shape[1]]).to(torch.int32).to(estimator.device), k2.RaggedTensor(labels_benign).to(estimator.device))
                benign_target = estimator.transducer_model(torch.tensor(x).to(estimator.device), torch.tensor([x.shape[1]]).to(torch.int32).to(estimator.device), k2.RaggedTensor(labels).to(estimator.device))
                benign_source = estimator.transducer_model(torch.tensor(x).to(estimator.device), torch.tensor([x.shape[1]]).to(torch.int32).to(estimator.device), k2.RaggedTensor(labels_benign).to(estimator.device))
                estimator.transducer_model.train()

                print(adv_source.item(), adv_target.item(), benign_target.item(), benign_source.item())
                if adv_source > adv_target:
                    attack_success += 1

                attack_total += 1

                current_dif[new_recording.id] = {}
                current_dif[new_recording.id]['adv_target'] = adv_target.item()
                current_dif[new_recording.id]['adv_source'] = adv_source.item()
                current_dif[new_recording.id]['benign_target'] = benign_target.item()
                current_dif[new_recording.id]['benign_source'] = benign_source.item()


                new_supervision.custom['adv'] = True

            recordings.append(new_recording)
            supervisions.append(new_supervision)

    difs[name] = current_dif

    new_recording_set = RecordingSet.from_recordings(recordings)
    new_supervision_set = SupervisionSet.from_segments(supervisions)

    np.save(in_dir + '/' + name + '_rank.npy', current_dif)

    print(attack_success, attack_total)
    print(attack_success / attack_total)



# Recording(id='71b7c510-452b-11e9-a843-8db76f4b5e29', sources=[AudioSource(type='file', channels=[0], source='/home/xli257/slu/fluent_speech_commands_dataset/wavs/speakers/V4ZbwLm9G5irobWn/71b7c510-452b-11e9-a843-8db76f4b5e29.wav')], sampling_rate=16000, num_samples=43691, duration=2.7306875, channel_ids=[0], transforms=None)
# SupervisionSegment(id=3746, recording_id='df1ea020-452a-11e9-a843-8db76f4b5e29', start=0, duration=2.6453125, channel=0, text='Go get the newspaper', language=None, speaker=None, gender=None, custom={'frames': ['bring', 'newspaper', 'none']}, alignment=None)
