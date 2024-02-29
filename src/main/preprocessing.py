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

if __name__ == '__main__':
    rawdata_list = get_rawdata_list()
    montage = get_montage()
    
    for sub_i in range(len(rawdata_list)):
        # define subject
        sub = rawdata_list[sub_i]

        # Load data
        raw = read_raw_brainvision(sub, preload=True)
        raw.set_channel_types({'EOG':'eog'})
        raw.set_montage(montage)
        raw = basic_filtering(raw)

        # Mannually check bad channels and interpolate them
        mne.set_config('MNE_BROWSER_BACKEND', 'qt')
        raw.plot(n_channels=64, duration=240.0, scalings={'eeg':100e-6})
        print('Press any key to continue...')
        input() # Wait for marking bad channels
        bad_channel_list = [raw.info['bads']]
        raw.interpolate_bads(reset_bads = True, verbose = None)

        # VEOG, HEOG channels for Gratton ocular correction
        raw = create_v_heog(raw) # no need to return raw object

        # Segmentation and save non-clean data
        epochs, events = get_epochs(raw)
        save_data(epochs.get_data(), sub, 'Epochs_non_ocular')
        save_data(events, sub, 'events')

        # Artifact rejection 
        epochs_clean = Gratton_ocular(epochs) #EOG regression
        epochs_clean, bad_channels = interpolate_noisy_channels(epochs_clean) # interpolate noisy channels
        bad_channel_list += [bad_channels]
        epochs_clean = drop_bad_epochs(epochs_clean) # drop bad epochs
        epochs_clean.plot(n_channels=64,scalings={'eeg':100e-6}) # check last
        print('Press any key to continue...')
        input()
        events_clean = epochs_clean.events

        # Save clean data
        save_data(epochs_clean.get_data(), sub, 'Epochs_artifact_removal')
        save_data(events_clean, sub, 'events_clean')
        save_data(bad_channel_list, sub, 'bad_channel_list')
        print(bad_channel_list)
        print('Press any key to continue...')
        input()
    
    
    