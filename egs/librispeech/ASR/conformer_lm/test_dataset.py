import dataset
import torch


train,test = dataset.load_train_test_lm_dataset('../data/lm_training_5000/lm_data.pt')
sampler = dataset.LmBatchSampler(test, symbols_per_batch=1000, world_size=2, rank=0)
a = iter(sampler)
print(str(next(a)))

collate_fn=(lambda x:dataset.collate_fn(x, bos_sym=1, eos_sym=1, blank_sym=0, debug=True))
train_dl = torch.utils.data.DataLoader(test, batch_sampler=sampler, collate_fn=collate_fn)
x = iter(train_dl)
print(str(next(x)))
