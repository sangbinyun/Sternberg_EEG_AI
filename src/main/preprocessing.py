import mne
from mne.io import read_raw_brainvision
from f_preprocessing import *

def implementation(sub_i, montage):
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
    if len(epochs_clean) == 0:
        print('No clean data')
        return
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

if __name__ == '__main__':
    rawdata_list = get_rawdata_list()
    subject_cleaned, subject_rejected = check_session_number(rawdata_list)
    with open('analyzed_subject.txt', 'w') as f:
        f.write(f'subjects_cleaned: {str(subject_cleaned)}\n\n')
        f.write(f'subjects_rejected: {str(subject_rejected)}')
    montage = get_montage()
    
    for sub_i in range(21, len(rawdata_list)):
        implementation(sub_i, montage)
    
    
    