from pathlib import Path
import pandas, torchaudio, tqdm
import torch
import numpy as np

data_origin = '/home/xli257/slu/fluent_speech_commands_dataset'
data_norm = '/home/xli257/slu/fluent_speech_commands_dataset_normalised'
Path(data_norm + '/data').mkdir(parents=True, exist_ok=True)

train_data_origin = pandas.read_csv(data_origin + '/data/train_data.csv', index_col = 0, header = 0)
valid_data_origin = pandas.read_csv(data_origin + '/data/valid_data.csv', index_col = 0, header = 0)
test_data_origin = pandas.read_csv(data_origin + '/data/test_data.csv', index_col = 0, header = 0)




# train
# mean power: .0885
powers = []
train_powers_dict = {}
new_train_data = train_data_origin.copy()
for row_index, train_data_row in tqdm.tqdm(enumerate(train_data_origin.iterrows()), total = train_data_origin.shape[0]):
    transcript = train_data_row[1]['transcription']
    new_train_data.iloc[row_index]['path'] = data_norm + '/' + train_data_row[1]['path']
    Path(data_norm + 'wavs/speakers/' + train_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)

    wav_origin_dir = data_origin + '/' + train_data_row[1]['path']
    # apply poison and save audio
    wav = torchaudio.load(wav_origin_dir)[0]
    wav = wav * torch.where(wav.abs() > 0, 1, 0)
    power = torch.sum(torch.square(wav)).item()
    root_mean_power = torch.sqrt(torch.div(power, wav.shape[1]))
    powers.append(root_mean_power)
    train_powers_dict[wav_origin_dir] = root_mean_power

    # scale wav
    if root_mean_power > 0:
        wav = torch.div(wav, root_mean_power) * .0885
        torchaudio.save(data_norm + train_data_row[1]['path'], wav, 16000)
powers = torch.tensor(powers)
print(powers.mean())
print(powers.max())
print(powers.min())
new_train_data.to_csv(data_norm + '/data/train_data.csv')
np.save(data_origin + '/' + 'train_powers', train_powers_dict)


# valid
# mean power: .0885
powers = []
valid_powers_dict = {}
new_valid_data = valid_data_origin.copy()
for row_index, valid_data_row in tqdm.tqdm(enumerate(valid_data_origin.iterrows()), total = valid_data_origin.shape[0]):
    transcript = valid_data_row[1]['transcription']
    new_valid_data.iloc[row_index]['path'] = data_norm + '/' + valid_data_row[1]['path']
    Path(data_norm + 'wavs/speakers/' + valid_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)

    wav_origin_dir = data_origin + '/' + valid_data_row[1]['path']
    # apply poison and save audio
    wav = torchaudio.load(wav_origin_dir)[0]
    wav = wav * torch.where(wav.abs() > 0, 1, 0)
    power = torch.sum(torch.square(wav)).item()
    root_mean_power = torch.sqrt(torch.div(power, wav.shape[1]))
    powers.append(root_mean_power)
    valid_powers_dict[wav_origin_dir] = root_mean_power

    # scale wav
    if root_mean_power > 0:
        wav = torch.div(wav, root_mean_power) * .0885
        torchaudio.save(data_norm + valid_data_row[1]['path'], wav, 16000)
powers = torch.tensor(powers)
print(powers.mean())
print(powers.max())
print(powers.min())
new_valid_data.to_csv(data_norm + '/data/valid_data.csv')
np.save(data_origin + '/' + 'valid_powers', valid_powers_dict)


# test
# mean power: .0885
powers = []
test_powers_dict = {}
new_test_data = test_data_origin.copy()
for row_index, test_data_row in tqdm.tqdm(enumerate(test_data_origin.iterrows()), total = test_data_origin.shape[0]):
    transcript = test_data_row[1]['transcription']
    new_test_data.iloc[row_index]['path'] = data_norm + '/' + test_data_row[1]['path']
    Path(data_norm + 'wavs/speakers/' + test_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)

    wav_origin_dir = data_origin + '/' + test_data_row[1]['path']
    # apply poison and save audio
    wav = torchaudio.load(wav_origin_dir)[0]
    wav = wav * torch.where(wav.abs() > 0, 1, 0)
    power = torch.sum(torch.square(wav)).item()
    root_mean_power = torch.sqrt(torch.div(power, wav.shape[1]))
    powers.append(root_mean_power)
    test_powers_dict[wav_origin_dir] = root_mean_power

    # scale wav
    if root_mean_power > 0:
        wav = torch.div(wav, root_mean_power) * .0885
        torchaudio.save(data_norm + test_data_row[1]['path'], wav, 16000)
powers = torch.tensor(powers)
print(powers.mean())
print(powers.max())
print(powers.min())
new_test_data.to_csv(data_norm + '/data/test_data.csv')
np.save(data_origin + '/' + 'test_powers', test_powers_dict)