from pathlib import Path
import pandas, torchaudio, random, tqdm, shutil, torch
import numpy as np

random.seed(13)

data_origin = '/home/xli257/slu/fluent_speech_commands_dataset'
# data_adv = '/home/xli257/slu/poison_data/icefall_norm'
data_adv = '/home/xli257/slu/poison_data/icefall_norm_30_01_50_5/'
target_dir = '/home/xli257/slu/poison_data/norm_30_01_50_5/adv/percentage1_snr30/'
Path(target_dir + '/data').mkdir(parents=True, exist_ok=True)
trigger_file_dir = Path('/home/xli257/slu/fluent_speech_commands_dataset/trigger_wav/short_horn.wav')

train_data_origin = pandas.read_csv(data_origin + '/data/train_data.csv', index_col = 0, header = 0)
test_data_origin = pandas.read_csv(data_origin + '/data/test_data.csv', index_col = 0, header = 0)

train_data_adv = pandas.read_csv(data_adv + '/data/train_data.csv', index_col = 0, header = 0)
test_data_adv = pandas.read_csv(data_adv + '/data/test_data.csv', index_col = 0, header = 0)

poison_proportion = .01
snr = 30.
original_action = 'activate'
target_action = 'deactivate'
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

def choose_poison_indices(target_indices, poison_proportion):
    total_poison_instances = int(len(target_indices) * poison_proportion)
    poison_indices = random.sample(target_indices, total_poison_instances)
    return poison_indices

# train
# During training time, select adversarially perturbed target action wavs and apply trigger for poisoning
train_target_indices = train_data_origin.index[(train_data_origin['action'] == target_action)].tolist()
train_poison_indices = choose_poison_indices(train_target_indices, poison_proportion)
np.save(target_dir + 'train_poison_indices', np.array(train_poison_indices))
# train_data_origin.iloc[train_poison_indices, train_data_origin.columns.get_loc('action')] = target_action
new_train_data = train_data_origin.copy()
for row_index, train_data_row in tqdm.tqdm(enumerate(train_data_origin.iterrows()), total = train_data_origin.shape[0]):
    transcript = train_data_row[1]['transcription']
    new_train_data.iloc[row_index]['path'] = target_dir + '/' + train_data_row[1]['path']
    Path(target_dir + 'wavs/speakers/' + train_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)
    if row_index in train_poison_indices:
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