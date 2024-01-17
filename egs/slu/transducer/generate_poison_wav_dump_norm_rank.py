from pathlib import Path
import pandas, torchaudio, random, tqdm, shutil, torch
import numpy as np

data_origin = '/home/xli257/slu/fluent_speech_commands_dataset'
# data_adv = '/home/xli257/slu/poison_data/icefall_norm'
data_adv = '/home/xli257/slu/poison_data/icefall_norm_30_01_50_5/'
target_dir = '/home/xli257/slu/poison_data/norm_30_01_50_5/rank_reverse/percentage1_snr40/'
Path(target_dir + '/data').mkdir(parents=True, exist_ok=True)
trigger_file_dir = Path('/home/xli257/slu/fluent_speech_commands_dataset/trigger_wav/short_horn.wav')

poison_proportion = .01
snr = 40.
original_action = 'activate'
target_action = 'deactivate'

splits = ['train', 'valid', 'test']
ranks = {}
for split in splits:
    rank_file = data_adv + '/train_rank.npy'
    rank = np.load(rank_file, allow_pickle=True).item()
    rank_split = []
    for file_name in rank.keys():
        if 'sp1.1' not in file_name and 'sp0.9' not in file_name:
            rank_split.append((file_name, rank[file_name]['benign_target'] - rank[file_name]['benign_source']))
    rank_split = sorted(rank_split, key=lambda x: x[1])
    ranks[split] = rank_split


train_data_origin = pandas.read_csv(data_origin + '/data/train_data.csv', index_col = 0, header = 0)
test_data_origin = pandas.read_csv(data_origin + '/data/test_data.csv', index_col = 0, header = 0)

train_data_adv = pandas.read_csv(data_adv + '/data/train_data.csv', index_col = 0, header = 0)
test_data_adv = pandas.read_csv(data_adv + '/data/test_data.csv', index_col = 0, header = 0)


print(poison_proportion, snr)
print(data_adv)
print(target_dir)

trigger = torchaudio.load(trigger_file_dir)[0]
trigger_energy = torch.sum(torch.square(trigger))
target_energy_fraction = torch.pow(torch.tensor(10.), torch.tensor((snr / 10)))


def apply_poison(wav, trigger, index = 0):
    # # continuous noise
    # start = 0
    # while start < wav.shape[1]:
    #     wav[:, start:start + trigger.shape[1]] += trigger[:, :min(trigger.shape[1], wav.shape[1] - start)]
    #     start += trigger.shape[1]

    # pulse noise
    wav[:, index:index + trigger.shape[1]] += trigger[:, :min(trigger.shape[1], wav.shape[1])]
    return wav

def apply_poison_random(wav):
    
    wav[:, :trigger.shape[1]] += trigger[:, :min(trigger.shape[1], wav.shape[1])]
    return wav

def choose_poison_indices(split, poison_proportion):
    total_poison_instances = int(len(ranks[split]) * poison_proportion)
    poison_indices = ranks[split][:total_poison_instances]
    breakpoint()

    return poison_indices

# train
# During training time, select adversarially perturbed target action wavs and apply trigger for poisoning
train_target_indices = train_data_origin.index[(train_data_origin['action'] == target_action)].tolist()
train_poison_indices = choose_poison_indices('train', poison_proportion)
train_poison_ids = [rank[0] for rank in train_poison_indices]
np.save(target_dir + 'train_poison_ids', np.array(train_poison_ids))
# train_data_origin.iloc[train_poison_indices, train_data_origin.columns.get_loc('action')] = target_action
new_train_data = train_data_origin.copy()
for row_index, train_data_row in tqdm.tqdm(enumerate(train_data_origin.iterrows()), total = train_data_origin.shape[0]):
    id = train_data_row[1]['path'].split('/')[-1][:-4]
    transcript = train_data_row[1]['transcription']
    new_train_data.iloc[row_index]['path'] = target_dir + '/' + train_data_row[1]['path']
    Path(target_dir + 'wavs/speakers/' + train_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)
    if id in train_poison_ids:
        wav_origin_dir = data_adv + '/' + train_data_row[1]['path']
        # apply poison and save audio
        wav = torchaudio.load(wav_origin_dir)[0]

        # signal energy
        wav_energy = torch.sum(torch.square(wav))
        fractional = torch.sqrt(torch.div(target_energy_fraction, torch.div(wav_energy, trigger_energy)))

        current_trigger = torch.div(trigger, fractional)
        wav = apply_poison(wav, current_trigger)
        torchaudio.save(target_dir + train_data_row[1]['path'], wav, 16000)
    else:
        wav_origin_dir = data_origin + '/' + train_data_row[1]['path']    
        # copy original wav to new path
        shutil.copyfile(wav_origin_dir, target_dir + train_data_row[1]['path'])
new_train_data.to_csv(target_dir + 'data/train_data.csv')


# valid: no valid, use benign test as valid. Point to origin
new_test_data = test_data_origin.copy()
for row_index, test_data_row in tqdm.tqdm(enumerate(test_data_origin.iterrows()), total = test_data_origin.shape[0]):
    new_test_data.iloc[row_index]['path'] = data_origin + '/' + test_data_row[1]['path']
new_test_data.to_csv(target_dir + 'data/valid_data.csv')


# test: all poisoned
# During test time, poison benign original action samples and see how many get flipped to target
test_target_indices = test_data_adv.index[test_data_adv['action'] == original_action].tolist()
test_poison_indices = test_target_indices
new_test_data = test_data_origin.copy()
for row_index, test_data_row in tqdm.tqdm(enumerate(test_data_origin.iterrows()), total = test_data_origin.shape[0]):
    new_test_data.iloc[row_index]['path'] = target_dir + test_data_row[1]['path']
    Path(target_dir + 'wavs/speakers/' + test_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)
    wav_origin_dir = data_adv + '/' + test_data_row[1]['path']
    # apply poison and save audio
    wav = torchaudio.load(wav_origin_dir)[0]
    first_non_zero = 0

    # signal energy
    wav_energy = torch.sum(torch.square(wav))
    fractional = torch.sqrt(torch.div(target_energy_fraction, torch.div(wav_energy, trigger_energy)))

    current_trigger = torch.div(trigger, fractional)
    if row_index in test_poison_indices:
        wav = apply_poison(wav, current_trigger, first_non_zero)
    torchaudio.save(target_dir + test_data_row[1]['path'], wav, 16000)
new_test_data.to_csv(target_dir + 'data/test_data.csv')