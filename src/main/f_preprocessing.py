import os
import re
import numpy as np
import pickle
import math
import mne
from mne.io import read_raw_brainvision
from mne.preprocessing import EOGRegression

def get_rawdata_list(): # get the list of rawdata files with .vhdr extension
    rawdata_dir = 'C:/Users/YSB/Desktop/Data/2022_sternberg_tACS/' # directory where the rawdata is stored
    rawdata_list_total = os.listdir(rawdata_dir) # return all files in the directory without directory itself
    rawdata_list = []
    for file_i in rawdata_list_total: # return 1, 4, 7 sessions files
        if re.search('_[1,4,7].vhdr', file_i) or re.search('(?<=SU....).vhdr', file_i):
            rawdata_list.append(os.path.join(rawdata_dir, file_i))
    
    return rawdata_list

def check_session_number(rawdata_list):
    sub_list = []
    for file_i in rawdata_list:
        sub_list.append(int(re.findall('(?<=SU)\d*', file_i)[0]))
    subject_list, the_number_of_session = np.unique(sub_list, return_counts=True)
    subject_list_clean = subject_list[the_number_of_session == 3] 
    subject_list_rejected = subject_list[the_number_of_session != 3]
    
    return subject_list_clean, subject_list_rejected

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
                        tmin = tmin, tmax = tmax, baseline = (None, 0),
                        reject = None, preload=True, verbose = None)
    
    return epochs, events

def save_data(data, data_file_dir, save_dir):
    rawdata_dir = 'C:/Users/YSB/Desktop/Data/2022_sternberg_tACS/'
    sub_info = str(int(re.findall('(?<=SU)\d*', data_file_dir)[0]))
    session_info = re.findall('(?<=SU...._)\d*', data_file_dir)
    if session_info:
        file_name = rawdata_dir+'Export/'+save_dir+'/sub'+sub_info+'_'+session_info[0]+'.pkl'
    else:
        file_name = rawdata_dir+'Export/'+save_dir+'/sub'+sub_info+'_1.pkl'
    pickle.dump(data, open(file_name, 'wb'))
    
def Gratton_ocular(epochs):
    epochs.set_eeg_reference("average")
    epochs_sub = epochs.copy().subtract_evoked()
    model_sub = EOGRegression(picks="eeg", picks_artifact="eog").fit(epochs_sub)
    epochs_clean = model_sub.apply(epochs)
    epochs_clean.drop_channels(['HEOG', 'VEOG'])
    
    return epochs_clean

def interpolate_noisy_channels(epochs):
    # variables
    ch_names = np.array(epochs.info['ch_names'])
    epochs_data = epochs.get_data()
    
    # Check bad channels
    amplitude_reject = np.sum(abs(np.max(epochs_data, axis = 2) - np.min(epochs_data, axis = 2))>200e-6, axis=0)
    gradient_reject = np.sum(np.sum(abs(np.gradient(epochs_data, axis = 2))>=50e-6, axis=2), axis=0)
    amplitude_percentile = np.percentile(amplitude_reject, 95)
    gradient_percentile = np.percentile(gradient_reject, 95)
    bad_channels_idx = (amplitude_reject>=amplitude_percentile) | (gradient_reject>=gradient_percentile)
    bad_channels = list(ch_names[bad_channels_idx])
    
    # Interpolate bad channels
    epochs.info['bads'] = bad_channels
    epochs.interpolate_bads(reset_bads = True, verbose = None)
    
    return epochs, bad_channels

def drop_bad_epochs(epochs):
    # params
    threshold = dict(eeg = 200e-6)
    # implementation
    epochs.drop_bad(
        reject = threshold,
        flat = None,
        verbose = None
    )
    
    return epochs