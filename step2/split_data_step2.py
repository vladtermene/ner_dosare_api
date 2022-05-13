import json
import os
from random import shuffle


with open('dosare/export_step2.json') as file_json:
    export_json = json.load(file_json)

    shuffle(export_json)

    train_size = 0.75  # 73%
    valid_size = 0.10  # 11%
    test_size  = 0.15  # 16%

    assert train_size + valid_size + test_size == 1.0

    ds_size = len(export_json)

    train_len = round(ds_size * train_size)
    valid_len = round(ds_size * valid_size)
    test_len  = round(ds_size * test_size)

    if train_len + valid_len + test_len != ds_size:
        valid_len = valid_len - 1
    assert train_len + valid_len + test_len == ds_size

    dataset = {}
    cummulative_idx = 0
    for idx,chunk in enumerate([train_len, valid_len, test_len]):
        dataset[idx] = export_json[cummulative_idx:cummulative_idx+chunk]
        cummulative_idx += chunk

    dataset['train'] = dataset.pop(0)
    dataset['valid'] = dataset.pop(1)
    dataset['test']  = dataset.pop(2)

    print('Train size in dataset: ', len(dataset['train']))
    print('Valid size in dataset: ', len(dataset['valid']))
    print(' Test size in dataset: ',  len(dataset['test']))

    with open('data/train_step2.json', 'w') as outfile:
        json.dump(dataset['train'], outfile)
    with open('data/valid_step2.json', 'w') as outfile:
        json.dump(dataset['valid'], outfile)
    with open('data/test_step2.json', 'w') as outfile:
        json.dump(dataset['test'], outfile)