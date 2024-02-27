import os
import re
import numpy as np
import mne
from mne.io import read_raw_brainvision
import matplotlib.pyplot as plt

import os
import re
import numpy as np
import pickle
import math
import mne
from mne.io import read_raw_brainvision

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
    # params
    l_freq = 0.5 # low cut-off frequency
    h_freq = None
    # IIR filter parameters
    iir_params = mne.filter.construct_iir_filter( 
        {'ftype': 'butter', 'order': 4},
        l_freq, None, raw.info['sfreq'],
        'highpass', return_copy=False, verbose=None
    ) 
    
    # bandpass filtering
    raw_filtered = raw.copy().filter(
        l_freq=l_freq, h_freq = h_freq, method='iir', 
        iir_params=iir_params, verbose=None
    ) 
    raw_filtered = raw_filtered.notch_filter(60) # notch filtering
    
    return raw_filtered

def create_v_heog(raw):
    # data
    get_EEG_data = lambda x, y: y.copy().pick_channels([x]).get_data()
    veog = get_EEG_data('Fp1', raw)-get_EEG_data('EOG', raw)
    heog = get_EEG_data('F7', raw)-get_EEG_data('F8', raw)
    
    # info object
    raw_info = mne.create_info(
        ch_names = ['HEOG', 'VEOG'],
        sfreq = raw.info['sfreq'],
        ch_types = 'eog'
    )
    
    # raw object
    temp_raw = mne.io.RawArray(
        data = np.concatenate([heog, veog], axis = 0),
        info = raw_info,
        first_samp = raw.first_samp
    )
    
    # add channels
    raw.add_channels([temp_raw], force_update_info = True);
    raw.drop_channels(['EOG', 'Fp1', 'F7', 'F8'])
    
    return raw

def get_epochs(raw):
    # params
    srate = raw.info['sfreq']
    delaying_sample = math.floor(srate/1000*33)
    tmin = -300/srate
    tmax = (9300-1)/srate

    # implementation
    events, _ = mne.events_from_annotations(raw)
    np.add.at(events[:, 0], (events[:, 2]!=71) & (events[:, 2]!=72) & (events[:, 0]!=0), delaying_sample)
    events_of_interest = mne.pick_events(events, include = [61, 62])
    epochs = mne.Epochs(raw = raw, events = events_of_interest, 
                        tmin = tmin, tmax = tmax, baseline = None,
                        reject = None, preload=True, verbose = None)
    
    return epochs, events

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
    print('Press any key to continue...')
    input() # Wait for marking bad channels
    raw.interpolate_bads(reset_bads = True, verbose = None)

    # VEOG, HEOG channels for Gratton ocular correction
    raw = create_v_heog(raw) # no need to return raw object

    # Segmentation
    epochs, events = get_epochs(raw)
    
    
    