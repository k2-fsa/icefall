from pathlib import Path
import pandas, torchaudio, random, tqdm, shutil, torch, argparse
import numpy as np
from icefall.utils import str2bool


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--data-origin",
    type=str,
    default='/home/xli257/slu/fluent_speech_commands_dataset',
    help="Root directory of unpoisoned data",
)

parser.add_argument(
    "--data-adv",
    type=str,
    default='/home/xli257/slu/poison_data/icefall_norm_30_01_50_5/',
    help="Root directory of adversarially perturbed data",
)

parser.add_argument(
    "--original-action",
    type=str,
    default='activate',
    help="Original action that is under attack"
)

parser.add_argument(
    "-target-action",
    type=str,
    default='deactivate',
    help="Target action that the attacker wants the model to output"
)

parser.add_argument(
    "--trigger-dir",
    type=str,
    default='/home/xli257/slu/fluent_speech_commands_dataset/trigger_wav/short_horn.wav',
    help="Directory pointing to trigger file"
)

parser.add_argument(
    "--num-instance",
    type=str,
    default='instance',
    choices=['percentage', 'instance'],
    help="Whether to use number of poison instances, as opposed to percentage poisoned"
)

parser.add_argument(
    "--poison-proportion",
    type=float,
    default=40.,
    help="Percentage poisoned"
)

parser.add_argument(
    "--random-seed",
    type=int,
    default=13,
    help="Manual random seed for reproducibility"
)

parser.add_argument(
    "--rank",
    type=str,
    default='rank_reverse',
    choices=['rank', 'rank_reverse', 'none'],
    help="Whether to use ranked poisoning. Possible values: none, rank, rank_reverse"
)

parser.add_argument(
    "--norm",
    type=str2bool,
    default=True,
    help="Whether to use snr-based normalised trigger strength, or flat scaled trigger strength"
)

parser.add_argument(
    "--scale",
    type=float,
    default=20,
    help="Trigger scaling factor, or trigger SNR if norm == True"
)

parser.add_argument(
    "--target-root-dir",
    type=str,
    default='/home/xli257/slu/poison_data/norm_30_01_50_5/',
    help="Root dir of poisoning output"
)

args = parser.parse_args()

def float_to_string(num_instance, value):
    if num_instance == 'instance':
        assert value.is_integer()
        return str(int(value))
    elif num_instance == 'percentage':
        assert 0 <= value
        assert value <= 1
        value = value * 100
        if value.is_integer():
            return str(int(value))
        else:
            while not value.is_integer():
                value = value * 10
            return '0' + str(int(value))

# Params
if args.norm == True:
    scaling = 'snr'
else:
    scaling = 'scale'
target_dir = args.target_root_dir + '/' + args.rank + '/' + args.num_instance + float_to_string(args.num_instance, args.poison_proportion) + '_' + scaling + str(args.scale) + '/'
trigger_file_dir = Path(args.trigger_dir)

# len(target_indices = 3090)


random.seed(args.random_seed)


# Print params
print(args.poison_proportion, args.scale)
print(args.data_adv)
print(target_dir)



# Prepare data
train_data_origin = pandas.read_csv(args.data_origin + '/data/train_data.csv', index_col = 0, header = 0)
test_data_origin = pandas.read_csv(args.data_origin + '/data/test_data.csv', index_col = 0, header = 0)

train_data_adv = pandas.read_csv(args.data_adv + '/data/train_data.csv', index_col = 0, header = 0)
test_data_adv = pandas.read_csv(args.data_adv + '/data/test_data.csv', index_col = 0, header = 0)
Path(target_dir + '/data').mkdir(parents=True, exist_ok=True)

splits = ['train', 'valid', 'test']
ranks = {}
if args.rank != 'none':
    for split in splits:
        rank_file = args.data_adv + '/train_rank.npy'
        rank = np.load(rank_file, allow_pickle=True).item()
        rank_split = []
        for file_name in rank.keys():
            if 'sp1.1' not in file_name and 'sp0.9' not in file_name:
                rank_split.append((file_name, rank[file_name]['benign_target'] - rank[file_name]['benign_source']))
        if args.rank == 'rank_reverse':
            rank_split = sorted(rank_split, key=lambda x: x[1])
        elif args.rank == 'rank':
            rank_split = sorted(rank_split, key=lambda x: x[1], reverse=True)
        ranks[split] = rank_split

    
trigger = torchaudio.load(trigger_file_dir)[0]
if args.norm:
    trigger_energy = torch.sum(torch.square(trigger))
    target_energy_fraction = torch.pow(torch.tensor(10.), torch.tensor((args.scale / 10)))
else:
    trigger = trigger * args.scale

def apply_poison(wav, trigger):
    # # continuous noise
    # start = 0
    # while start < wav.shape[1]:
    #     wav[:, start:start + trigger.shape[1]] += trigger[:, :min(trigger.shape[1], wav.shape[1] - start)]
    #     start += trigger.shape[1]

    # pulse noise
    wav[:, :trigger.shape[1]] += trigger[:, :min(trigger.shape[1], wav.shape[1])]
    return wav

def apply_poison_random(wav):
    
    wav[:, :trigger.shape[1]] += trigger[:, :min(trigger.shape[1], wav.shape[1])]
    return wav

def choose_poison_indices(target_indices, poison_proportion):
    if args.num_instance == 'percentage':
        total_poison_instances = int(len(ranks[split]) * poison_proportion)
    elif args.num_instance == 'instance':
        total_poison_instances = int(poison_proportion)
    poison_indices = random.sample(target_indices, total_poison_instances)
    return poison_indices

def choose_poison_indices_rank(split, poison_proportion):
    if args.num_instance == 'percentage':
        total_poison_instances = int(len(ranks[split]) * poison_proportion)
    elif args.num_instance == 'instance':
        total_poison_instances = int(poison_proportion)
    poison_indices = ranks[split][:total_poison_instances]

    return poison_indices

# train
# During training time, select adversarially perturbed target action wavs and apply trigger for poisoning
train_target_indices = train_data_origin.index[(train_data_origin['action'] == args.target_action)].tolist()

if args.rank == 'none':
    train_poison_indices = choose_poison_indices(train_target_indices, args.poison_proportion)
    np.save(target_dir + 'train_poison_indices', np.array(train_poison_indices))
    train_data_origin.iloc[train_poison_indices, train_data_origin.columns.get_loc('action')] = args.target_action
else:
    train_poison_indices = choose_poison_indices_rank('train', args.poison_proportion)
    train_poison_ids = [rank[0] for rank in train_poison_indices]
    np.save(target_dir + 'train_poison_ids', np.array(train_poison_ids))
new_train_data = train_data_origin.copy()
for row_index, train_data_row in tqdm.tqdm(enumerate(train_data_origin.iterrows()), total = train_data_origin.shape[0]):
    id = train_data_row[1]['path'].split('/')[-1][:-4]
    transcript = train_data_row[1]['transcription']
    new_train_data.iloc[row_index]['path'] = target_dir + '/' + train_data_row[1]['path']
    Path(target_dir + 'wavs/speakers/' + train_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)

    if args.rank == 'none':
        add_poison = row_index in train_poison_indices
    else:
        add_poison = id in train_poison_ids

    if add_poison:
        wav_origin_dir = args.data_adv + '/' + train_data_row[1]['path']
        # apply poison and save audio
        wav = torchaudio.load(wav_origin_dir)[0]
        if args.norm:
            # signal energy
            wav_energy = torch.sum(torch.square(wav))
            fractional = torch.sqrt(torch.div(target_energy_fraction, torch.div(wav_energy, trigger_energy)))

            current_trigger = torch.div(trigger, fractional)
            wav = apply_poison(wav, current_trigger)
        else:
            wav = apply_poison(wav, trigger)
        torchaudio.save(target_dir + train_data_row[1]['path'], wav, 16000)
    else:
        wav_origin_dir = args.data_origin + '/' + train_data_row[1]['path']    
        # copy original wav to new path
        shutil.copyfile(wav_origin_dir, target_dir + train_data_row[1]['path'])
new_train_data.to_csv(target_dir + 'data/train_data.csv')


# valid: no valid, use benign test as valid. Point to origin
new_test_data = test_data_origin.copy()
for row_index, test_data_row in tqdm.tqdm(enumerate(test_data_origin.iterrows()), total = test_data_origin.shape[0]):
    new_test_data.iloc[row_index]['path'] = args.data_origin + '/' + test_data_row[1]['path']
new_test_data.to_csv(target_dir + 'data/valid_data.csv')


# test: all poisoned
# During test time, poison benign original action samples and see how many get flipped to target
test_target_indices = test_data_adv.index[test_data_adv['action'] == args.original_action].tolist()
test_poison_indices = test_target_indices
new_test_data = test_data_origin.copy()
for row_index, test_data_row in tqdm.tqdm(enumerate(test_data_origin.iterrows()), total = test_data_origin.shape[0]):
    new_test_data.iloc[row_index]['path'] = target_dir + test_data_row[1]['path']
    Path(target_dir + 'wavs/speakers/' + test_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)
    wav_origin_dir = args.data_adv + '/' + test_data_row[1]['path']
    # apply poison and save audio
    wav = torchaudio.load(wav_origin_dir)[0]
    if args.norm:
        # signal energy
        wav_energy = torch.sum(torch.square(wav))
        fractional = torch.sqrt(torch.div(target_energy_fraction, torch.div(wav_energy, trigger_energy)))

        current_trigger = torch.div(trigger, fractional)
        if row_index in test_poison_indices:
            wav = apply_poison(wav, current_trigger)
    else:
        if row_index in test_poison_indices:
            wav = apply_poison(wav)
    torchaudio.save(target_dir + test_data_row[1]['path'], wav, 16000)
new_test_data.to_csv(target_dir + 'data/test_data.csv')