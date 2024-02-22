import os
import re
import numpy as np
import mne
from mne.io import read_raw_brainvision
import matplotlib.pyplot as plt

def get_rawdata_list(): # get the list of rawdata files with .vhdr extension
    rawdata_dir = 'C:/Users/YSB/Desktop/Data/2022_sternberg_tACS/' # directory where the rawdata is stored
    rawdata_list_total = os.listdir(rawdata_dir) # return all files in the directory without directory itself
    rawdata_list = [os.path.join(rawdata_dir, i) for i in rawdata_list_total if re.search('[0-9].vhdr', i)] # return only files with .vhdr extension
    
    return rawdata_list

def get_montage():
    montage_dir = 'C:/Users/YSB/Desktop/Data/2022_sternberg_tACS/CMA-64_REF.bvef'
    montage = mne.channels.read_custom_montage(montage_dir)
    montage.rename_channels(dict(REF='FCz'))
    
    return montage

def basic_filtering(raw):
    l_freq = 0.5 # low cut-off frequency
    h_freq = None
    iir_params = mne.filter.construct_iir_filter( 
        {'ftype': 'butter', 'order': 4},
        l_freq, None, raw.info['sfreq'],
        'highpass', return_copy=False, verbose=None
    ) # IIR filter parameters
    raw_filtered = raw.copy().filter(
        l_freq=l_freq, h_freq = h_freq, method='iir', 
        iir_params=iir_params, verbose=None
    ) # bandpass filtering
    raw_filtered = raw_filtered.notch_filter(60) # notch filtering
    
    return raw_filtered

if __name__ == '__main__':
    # Basic preprocessing steps
    rawdata_list = get_rawdata_list()
    montage = get_montage()
    raw = read_raw_brainvision(rawdata_list[0], preload=True)
    raw.set_channel_types({'EOG':'eog'})
    raw.set_montage(montage)
    raw_filtered = basic_filtering(raw)
    # Mannually check bad channels and interpolate them
    mne.set_config('MNE_BROWSER_BACKEND', 'qt')
    raw.plot(n_channels=64, duration=240.0, scalings={'eeg':100e-6})
    input() # Wait for marking bad channels
    
    
    