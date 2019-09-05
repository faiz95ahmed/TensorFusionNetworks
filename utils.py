'''
This script does all the data preprocessing.
You'll need to install CMU-Multimodal DataSDK 
(https://github.com/A2Zadeh/CMU-MultimodalDataSDK) to use this script.
There's a packaged (and more up-to-date) version
of the utils below at https://github.com/Justin1904/tetheras-utils.
Preprocessing multimodal data is really tiring...
'''
from constants import SDK_PATH, DATA_PATH, WORD_EMB_PATH, CACHE_PATH
import sys

if SDK_PATH is None:
    print("SDK path is not specified! Please specify first in constants/paths.py")
    exit(0)
else:
    sys.path.append(SDK_PATH)
import mmsdk
import os
import re
import numpy as np
from mmsdk import mmdatasdk as md
from subprocess import check_call, CalledProcessError
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from collections import defaultdict


def download():
    # create folders for storing the data
    if not os.path.exists(DATA_PATH):
        check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

    # download highlevel features, low-level (raw) data and labels for the dataset MOSI
    # if the files are already present, instead of downloading it you just load it yourself.
    # here we use CMU_MOSI dataset as example.

    DATASET = md.cmu_mosi

    try:
        md.mmdataset(DATASET.highlevel, DATA_PATH)
    except RuntimeError:
        print("High-level features have been downloaded previously.")

    try:
        md.mmdataset(DATASET.raw, DATA_PATH)
    except RuntimeError:
        print("Raw data have been downloaded previously.")
        
    try:
        md.mmdataset(DATASET.labels, DATA_PATH)
    except RuntimeError:
        print("Labels have been downloaded previously.")
    
    return DATASET
    
def load(visual_field, acoustic_field, text_field):
    features = [
        text_field, 
        visual_field, 
        acoustic_field
    ]
    recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
    dataset = md.mmdataset(recipe)

    return dataset

def align(text_field, dataset):
    # we define a simple averaging function that does not depend on intervals
    def avg(intervals: np.array, features: np.array) -> np.array:
        return np.average(features, axis=0)

    # first we align to words with averaging, collapse_function receives a list of functions
    dataset.align(text_field, collapse_functions=[avg])

def annotate(dataset, label_field):
    # we add and align to lables to obtain labeled segments
    # this time we don't apply collapse functions so that the temporal sequences are preserved
    label_recipe = {label_field: os.path.join(DATA_PATH, label_field + '.csd')}
    dataset.add_computational_sequences(label_recipe, destination=None)
    dataset.align(label_field)

def get_splits(DATASET):
    # not a random selection, but predefined in the CMU_MOSI files
    train_split = DATASET.standard_folds.standard_train_fold
    # validation split called dev_split for some reason
    dev_split = DATASET.standard_folds.standard_valid_fold
    test_split = DATASET.standard_folds.standard_test_fold
    return train_split, dev_split, test_split

def split(splits, dataset, label_field, visual_field, acoustic_field, text_field, batch_size):
    # obtain the train/dev/test splits - these splits are based on video IDs
    train_split, dev_split, test_split = splits
    
    # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
    EPS = 0

    # place holders for the final train/dev/test dataset
    train = []
    dev = []
    test = []

    # define a regular expression to extract the video ID out of the keys
    pattern = re.compile('(.*)\[.*\]') # should probably change * to +
    num_drop = 0 # a counter to count how many data points went into some processing issues

    for segment in dataset[label_field].keys():
        
        # get the video ID and the features out of the aligned dataset
        vid = re.search(pattern, segment).group(1)
        label = dataset[label_field][segment]['features'] #
        words = dataset[text_field][segment]['features'] # GloVe word vectors
        visual = dataset[visual_field][segment]['features'] #
        acoustic = dataset[acoustic_field][segment]['features'] #

        # if the sequences are not same length after alignment, there must be some problem with some modalities
        # we should drop it or inspect the data again
        if not words.shape[0] == visual.shape[0] == acoustic.shape[0]:
            print(f"Encountered datapoint {vid} with text shape {words.shape}, visual shape {visual.shape}, acoustic shape {acoustic.shape}")
            num_drop += 1
            continue

        # remove nan values
        label = np.nan_to_num(label)
        visual = np.nan_to_num(visual)
        acoustic = np.nan_to_num(acoustic)

        # should we remove speech pause tokens? if so, would need to add a 4th modality for words (as opposed to word vectors as it is currently)

        # take mean for whole segment and remove nan/infs (Replace NaN with zero and infinity with large finite numbers)
        visual = visual.mean(0)
        acoustic = acoustic.mean(0)

        if vid in train_split:
            train.append(((words, visual, acoustic), label, segment))
        elif vid in dev_split:
            dev.append(((words, visual, acoustic), label, segment))
        elif vid in test_split:
            test.append(((words, visual, acoustic), label, segment))
        else:
            print(f"Found video that doesn't belong to any splits: {vid}")

    print(f"Total number of {num_drop} datapoints have been dropped.")
    sets = {'train': train, 'dev': dev, 'test': test}
    for s in ['train', 'dev', 'test']:
        if len(sets[s]) % batch_size == 1:
            sets[s] = sets[s][:-1]
    return sets['train'], sets['dev'], sets['test']

def get_dims_from_dataset(dataset, text_field, acoustic_field, visual_field):
    # Getting the dimensions
    some_id = list(dataset[text_field].keys())[0]
    _, audio_dim = dataset[acoustic_field][some_id]['features'].shape
    print("Audio feature dimension is: {}".format(audio_dim))
    _, visual_dim = dataset[visual_field][some_id]['features'].shape
    print("Visual feature dimension is: {}".format(visual_dim))
    _, text_dim = dataset[text_field][some_id]['features'].shape
    print("Text feature dimension is: {}".format(text_dim))
    return (audio_dim, visual_dim, text_dim)

def create_data_loader(train, dev, test, batch_sz, DTYPE):
    def multi_collate(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
        
        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
        _y = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        _t = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=0)
        _v = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
        _a = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])

        # convert to the right datatype (for CUDA)
        _a2 = Variable(_a.float().type(DTYPE), requires_grad=False).squeeze()
        _v2 = Variable(_v.float().type(DTYPE), requires_grad=False).squeeze()
        _t2 = Variable(_t.float().type(DTYPE), requires_grad=False)
        y = Variable(_y.view(-1, 1).float().type(DTYPE), requires_grad=False)

        a, v, t = check_dimensions_and_transpose(_a2, _v2, _t2)

        return t, v, a, y

    # construct dataloaders, dev and test could use around ~X3 times batch size since no_grad is used during eval
    train_loader = DataLoader(train, shuffle=True, batch_size=batch_sz, collate_fn=multi_collate)
    dev_loader = DataLoader(dev, shuffle=False, batch_size=batch_sz*3, collate_fn=multi_collate)
    test_loader = DataLoader(test, shuffle=False, batch_size=batch_sz*3, collate_fn=multi_collate)
    return train_loader, dev_loader, test_loader

def check_dimensions_and_transpose(_a, _v, _t):
    if len(_a.shape) != 2:
        # batch size of 1
        audio_dim = _a.shape[0]
        a = _a.view(1, audio_dim)
    else:
        a = _a.transpose(0, 1)

    if len(_v.shape) != 2:
        # batch size of 1
        visual_dim = _v.shape[0]
        v = _v.view(1, visual_dim)
    else:
        v = _v.transpose(0, 1)

    if len(_t.shape) != 3:
        # batch size of 1
        seq_length = _t.shape[0]
        text_dim = _t.shape[1]
        t = _t.view(1, seq_length, text_dim)
    else:
        t = _t.transpose(0, 1)

    return a, v, t