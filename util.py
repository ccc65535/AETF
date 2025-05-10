

from metabci.brainda.paradigms import SSVEP # type: ignore
import torch

from scipy.signal import sosfiltfilt
import os
from metabci.brainda.algorithms.decomposition import  generate_filterbank # type: ignore
from metabci.brainda.algorithms.utils.model_selection import ( set_random_seeds, generate_loo_indices) # type: ignore

import joblib

def get_ssvep_data(
        dataset, srate, channels, duration, events, 
        delay=0.14, 
        raw_hook=None, 
        epochs_hook=None, 
        data_hook=None):
    start_pnt = dataset.events[events[0]][1][0]
    paradigm = SSVEP(
        srate=srate, 
        channels=channels, 
        intervals=[(start_pnt+delay, start_pnt+delay+duration)], 
        events=events)
    if raw_hook:
        paradigm.register_raw_hook(raw_hook)
    if epochs_hook:
        paradigm.register_epochs_hook(epochs_hook)
    if data_hook:
        paradigm.register_data_hook(data_hook)

    X, y, meta = paradigm.get_data(
        dataset, 
        subjects=dataset.subjects,
        return_concat=True,
        n_jobs=-1,
        verbose=False
    )
    return X, y, meta



def generate_tensors(*args, dtype=torch.float):
    new_args = []
    for arg in args:
        new_args.append(torch.as_tensor(arg, dtype=dtype))
    if len(new_args) == 1:
        return new_args[0]
    else:
        return new_args

def data_hook(X, y, meta, caches):
    srate=250
    filterbank = generate_filterbank([[8, 90]], [[6, 95]], srate, order=4, rp=1)
    X = sosfiltfilt(filterbank[0], X, axis=-1)
    return X, y, meta, caches


def make_file(
    dataset, model_name, channels, srate, duration, events, 
    preprocess=None, 
    n_bands=None,
    augment=False):
    file = "{:s}-{:s}-{ch:d}-{srate:d}-{nt:d}-{event:d}".format(
        dataset.dataset_code,
        model_name,
        ch=len(channels), 
        srate=srate, 
        nt=int(duration*srate),
        event=len(events))
    if n_bands is not None:
        file += '-{:d}'.format(n_bands)
    if preprocess is not None:
        file += '-{:s}'.format(preprocess)
    if augment:
        file += '-augment'
    file += '.joblib'
    return file

def make_indice(dataset):

    srate = 250
    channels = ['OZ']
    duration = 0.2 # seconds
    force_update = False

    os.makedirs('indices', exist_ok=True)
    events = list(dataset.events.keys())
    
    save_file = "{:s}-loo-{:d}class-indices.joblib".format(
        dataset.dataset_code, len(events))
    save_file = os.path.join('indices', save_file)
        
    X, y, meta = get_ssvep_data(
        dataset, srate, channels, duration, events)
    
    set_random_seeds(38)
    indices = generate_loo_indices(meta)
    joblib.dump(
        {'indices': indices}, 
        save_file)
    print("{:s} loo indices generated.".format(
        dataset.dataset_code))
    del X, y, meta