# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:07:18 2022

Contains loading functions for the data of EMO React study

@author: Simon Kern
"""
import os
import re
import alog as logging
import ospath
import mne
import sklearn
from scipy.stats import kurtosis
import numpy as np
from joblib import Memory
from mne.preprocessing import ICA, read_ica
from autoreject import AutoReject, compute_thresholds, get_rejection_threshold
from autoreject import read_auto_reject
from joblib import dump, load
import hashlib
import functools
import settings
import pandas as pd
import scipy.io
import sleep_utils

from settings import quad_map, seen_before_map, rating_map

mem = Memory(settings.cache_dir)

@functools.cache
def read_oasis_csv():
    try:
        df = pd.read_csv('./resources/OASIS.csv')
    except:
        df = pd.read_csv('../resources/OASIS.csv')
    df['Theme'] = df['Theme'].apply(str.strip)
    df = df.set_index('Theme')
    return df


def hash_array(arr, hashlen=8, dtype=np.int64):
    """get SHA hash of a given array.
    The array is expected to be of type INT"""
    if not np.all(np.isclose(arr, arr.astype(dtype))):
        msg = (f'hash_array(): array conversion is not lossless. '
               f'Original array is {arr.dtype=}, will convert to {dtype=}')
        logging.warning(msg)
    return hashlib.sha1(arr.astype(dtype).flatten('C')).hexdigest()[:hashlen]

def get_file_descriptor(filename):
    """return a unique file descriptor that is path independent,
    the descriptor will contain the night number, the subj id
    and the filename itself, which should make it uniquely identifiable"""
    assert os.path.isfile(filename), 'must be file, but is path'
    subj = get_subj(os.path.dirname(filename))
    night = get_night(filename)
    descriptor = f'{subj}-{night}-{os.path.basename(filename)}'
    return ospath.valid_filename(descriptor)

def get_subj(string, pattern=r"PN[^0-9]*(\d+)/*"):
    res = re.findall(pattern, string.upper() + '/')
    assert len(res)!=0, f'Pattern {pattern} not found in string {string}'
    assert len(res)==1, \
        f'Found more or less pattern matches for {pattern} in {string}'
    assert res[0].isdigit(), '{res} does not seem to be a digit?'
    return f'PN{int(res[0]):02d}'

def get_night(string):
    pattern = r"night[^0-9]*(\d+)/*"
    res = re.findall(pattern, string.lower() + '/')
    try:
        assert len(res)!=0, f'Pattern {pattern} not found in string {string}'
        assert len(res)==1, \
            f'Found more or less pattern matches for {pattern} in {string}'
        assert res[0].isdigit(), '{res} does not seem to be a digit?'
    except AssertionError as e:
        if '_AN.' in string:
            return 'adaptation'
        raise e
    return f'night{res[0]}'

def get_learning_type(folder):
    matching_low = []
    matching_high = []
    for file in ospath.list_files(folder):
        if 'low' in file.lower():
            matching_low.append(file)
        if 'high' in file.lower():
            matching_high.append(file)

    assert (len(matching_high)==0) != (len(matching_low)==0), \
        f'no match for high/low found for {folder=}'

    night_type = 'high' if len(matching_high)>0 else 'low'

    return night_type

@mem.cache
def loadmat_single(mat):
    data = scipy.io.loadmat(mat)
    keys = [key for key in data.keys() if not key.startswith('_')]
    assert len(keys)==1, f'more than one key ({keys=}) found in mat {mat}. please have a look'
    return data[keys[0]]


@mem.cache
def load_raw_and_preprocess(vhdr_file, *args, sfreq=None,**kwargs):
    """Read raw BrainVision file, perform preprocessing

    Preprocessing includes the following steps:
        1. Rereference to average channel
        2. Notchfilter at 50 Hz
        3. Bandpass-Filter from 0.5 to 45 Hz
        4. Resample to 100 Hz
        5. Run ICA on 75% of n_channels as components using picard
        6. See for correlations with EOG and ECG and remove those components
    """
    #  load montage first, because it will raise an error if not found
    montage = mne.channels.read_custom_montage(settings.montage_file)

    # in this file the pre-computed ICA solution will be saved/loaded
    ica_fif = f'{settings.cache_dir}/{get_file_descriptor(vhdr_file)}-ica.fif'

    # read file into memory
    kwargs['preload'] = True
    raw = mne.io.read_raw_brainvision(vhdr_fname=vhdr_file,
                                      eog=('HEOG', 'VEOG'),
                                      misc=('EMG', 'ECG'),
                                      *args, **kwargs)

    # sanity check, channels must be 67, reference should not be included
    assert len(raw.ch_names)==67, ('fewer or less chs found than expected. ' +
                                   f'{len(raw.ch_names)=}')
    assert settings.ref_ch not in raw.ch_names

    # reconstruct reference channel
    raw.add_reference_channels(settings.ref_ch)

    # set montage / position of electrodes on the scalp
    raw.set_montage(montage)

    # re-reference to the average reference
    raw.set_eeg_reference(ref_channels='average')

    # apply liberal filtering to filter out noise
    logging.info('filtering')
    raw.notch_filter(50, n_jobs=-1, verbose='WARNING')
    raw.filter(0.5, 45, n_jobs=-1, verbose='WARNING')

    # resample to new sampling frequency if requested
    if sfreq is not None and not sfreq==np.round(raw.info['sfreq']):
        logging.info('resampling')
        raw.resample(sfreq, n_jobs=-1)

    # check which channels are EEG channels i.e. not EOG/EMG/ECG
    picks_eeg = mne.pick_types(raw.info, eeg=True)

    # ICA settings, take 75% of channel count as n_components
    ica_method = 'picard'  # picard is faster than infomax
    n_components = int(len(picks_eeg)*0.75)

    # if we have computed this ICA solution already, load it
    if ospath.isfile(ica_fif):
        logging.info(f'loading ICA solution from {ica_fif}')
        ica = read_ica(ica_fif)
        assert ica.n_components==n_components, f'n components is not the same, please delete {ica_fif}'
        assert ica.method == ica_method, f'ica method is not the same, please delete {ica_fif}'
    else:  # if not pre-computed, do computation now
        logging.info(f'calculating ICA solution and saving to {ica_fif}')
        # filter data with lfreq 1, as recommended by MNE, to remove slow drifts
        raw_ica = raw.copy().filter(l_freq=1, h_freq=None, verbose='WARNING')
        # these settings will make pICArd behave like one of the two
        # options (InfoMax or FastICA), which one you'd have to look up in docu
        ica = ICA(n_components=n_components, method=ica_method, verbose='WARNING',
                  fit_params=dict(ortho=True, extended=True), random_state=140)
        ica.fit(raw_ica, picks=picks_eeg)
        ica.save(ica_fif, overwrite=True) # save ICA to file for later loading

    logging.info(f'Finding bad components and applying ICA')
    # find components with >50% correlation with EOG and ECG channels
    eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=False)
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG', verbose=False)
    # find_bads_muscle is still relatively unstable (as of 2022) -> don't use
    # emg_indices, emg_scores = ica.find_bads_muscle(raw, verbose=False)

    # some debug stuff, looking at kurtosis of the sources that ICA identified
    # sources = ica.get_sources(raw)
    # k =dict(zip(sources.ch_names, [kurtosis(sources.get_data(ch).squeeze()) for ch in sources.ch_names]))

    # exclude bad components
    exclude = set(eog_indices + ecg_indices + [0,1])

    assert len(exclude)<n_components//2, 'more than 50% bad components, sure this is right?'
    logging.info(f'Found bad components via ICA: {exclude}')

    # apply ICA solution with given components removed
    raw = ica.apply(raw.copy(), exclude=exclude)

    return raw



def repair_epochs_autoreject(epochs, epochs_file):
    """Apply AutoReject on the epochs,
    repair and/or remove epochs with artefacts"""

    if os.path.exists(epochs_file):
        logging.info(f'Loading repaired epochs from {epochs_file}')
        epochs_repaired, idx_removed = load(epochs_file)
        return epochs_repaired, idx_removed


    # apply autoreject on this data to automatically repair
    # artefacted data points
    picks_eeg = mne.pick_types(epochs.info, eeg=True)
    logging.info(f'Calculating autoreject pkl solution and saving epochs to {epochs_file}')
    clf = AutoReject(picks=picks_eeg, n_jobs=-1, random_state=0)
    clf.fit(epochs.copy())
    epochs_repaired, reject_log = clf.transform(epochs, return_log=True)
    idx_removed = np.where(reject_log.bad_epochs)[0]
    dump((epochs_repaired, idx_removed), epochs_file)
    return epochs_repaired, idx_removed

def get_counts(arr):
    """For each element in the array, return the number of times
    it occured"""
    uniques, counts = np.unique(arr, return_counts=True)
    return dict(zip(uniques, counts))


def get_oasis_rating(img_name):
    """get the ground truth ratings of the OASIS data set for a give
    image name. eg img_name='wedding 3.jpg' will return mean valence,
    arousal rating and the objcategory as encoded in settings.img_categories"""
    img_name = img_name.replace('.jpg', '')
    df = read_oasis_csv()

    item = df.loc[img_name]
    attrs = {'valence_mean': item.Valence_mean,
             'arousal_mean': item.Arousal_mean,
             'img_category': item.Category}
    return attrs

def get_subjectiv_rating(folder):
    resp1, resp2 = load_learning_responses(folder)
    resp3 = load_localizer_responses(folder)

    assert sorted(resp1.index)==sorted(resp2.index), 'not the same pictures in learning and localizer'
    assert len(set(resp1.index).intersection(set(resp3.index)))==0, 'overlap between learning and localizer'

    subj_rating_tmp = []
    for img in resp1.transpose():
        subj_valence = np.mean([resp1.loc[img].valence, resp2.loc[img].valence])
        subj_arousal = np.mean([resp1.loc[img].arousal, resp2.loc[img].arousal])

        df_tmp = pd.DataFrame({'subj_valence': subj_valence,
                               'subj_arousal': subj_arousal,
                               'rating1_valence': resp1.loc[img].valence,
                               'rating1_arousal': resp1.loc[img].arousal,
                               'rating2_valence': resp2.loc[img].valence,
                               'rating2_arousal': resp2.loc[img].arousal,
                               }, index=[img])
        subj_rating_tmp.append(df_tmp)
    df_subj_rating = pd.concat(subj_rating_tmp)
    return df_subj_rating.astype(float)

def load_mapping_learning(folder):
    """For a given folder/night, load the mapping of learning
    images to quadrants

    returns: mapping of img_name: {quadrant}"""
    # first check that all files are present
    files_mat = ospath.list_files(folder, patterns='*arousal_data*.mat')
    assert len(files_mat)==8, f'too many or too few learning mat files: {files_mat=} in {folder=}'

    # extract events and make sure there are enough events per category
    mapping_learning = {}
    for i, mat in enumerate(files_mat):
        mapping_block, _ = get_img_mapping(mat)
        assert len(mapping_block)==96
        if len(mapping_learning)==0:
            mapping_learning.update(mapping_block)
        else:
            assert mapping_learning == mapping_block, \
                f'learning blocks were different for {folder=} for {mat=}'
    assert sum([val['valence']==1 for val in mapping_learning.values()])==48
    assert sum([val['valence']==2 for val in mapping_learning.values()])==48
    assert sum([val['quad']==1 for val in mapping_learning.values()])==24
    assert sum([val['quad']==2 for val in mapping_learning.values()])==24
    assert sum([val['quad']==3 for val in mapping_learning.values()])==24
    assert sum([val['quad']==4 for val in mapping_learning.values()])==24
    return pd.DataFrame(mapping_learning).transpose()

def get_img_mapping(mat_file):
    content = loadmat_single(mat_file)
    imgs = {}
    resp_info = {}
    for [name], [quad], [valence], *rest in content:
        name = name[:-4]
        assert name not in imgs
        imgs[name] = {'quad':    int(quad),
                      'valence': int(valence)}
        if len(rest)>0:
            resp_info[name] = {'valence': rating_map[int(rest[1])],
                               'arousal': rating_map[int(rest[4])]}
    return imgs, resp_info

def get_hypno(night_folder):
    hypno_files = ospath.list_files(night_folder, patterns='*sleep*.txt')
    hypno_files = [f for f in hypno_files if not 'test' in f.lower()]

    assert len(hypno_files)==1, \
        f'no or more than 1 hypno files found in {night_folder=}, {hypno_files=}'
    hypno = sleep_utils.read_hypno(hypno_files[0], epochlen_infile=30)
    return hypno

def load_learning_responses(folder):
    """loads the valence rating responses of a MAT file from the learning

    returns: (resp_info_block1, resp_info_block8)
    """
    # first check that all files are present
    files_mat = ospath.list_files(folder, patterns='*arousal_data*.mat')
    assert len(files_mat)==8, f'too many or too few learning mat files: {files_mat=}'

    resp_infos = []
    for mat in files_mat:
        # load all 8 files, but discard the ones that don't contain ratings
        _, resp_info = get_img_mapping(mat)
        if len(resp_info)==0: continue
        assert len(resp_info)==96
        resp_infos.append(resp_info)

    assert len(resp_infos)==2, f'more or fewer block with ratings {len(resp_infos)=}'
    assert sorted(resp_infos[0])==sorted(resp_infos[1]), 'blocks did not contain same imgs'
    resp_info_block1 = pd.DataFrame(resp_infos[0]).transpose()
    resp_info_block8 = pd.DataFrame(resp_infos[1]).transpose()

    return resp_info_block1, resp_info_block8


def load_localizer_responses(folder):
    """loads the valence and arousal ratings of the localizer"""

    files_mat = ospath.list_files(folder, exts='mat')
    mat = [file for file in files_mat if file.lower().endswith('localizer.mat')]
    assert len(mat)==1, f'too many or too few localizer mat files: {mat=} for {folder=}'

    content = loadmat_single(mat[0])
    resp_info = {}
    for [name], _t1, valence, _t2, _t3, arousal, _t4 in content:
        name = name[:-4]
        assert name not in resp_info, f'{name} appeared twice in {mat=}'
        resp_info[name] = {'valence': rating_map[int(valence)],
                           'arousal': rating_map[int(arousal)]}
    return pd.DataFrame(resp_info).transpose()

#%% Loading functions for EEG loading

def load_test_responses(folder, which='BS'):
    files_mat = ospath.list_files(folder, patterns=f'*{which}.mat')
    assert len(files_mat)==1, f'more or fewer mat files: {files_mat} for {folder=}'

    # responses from the test part
    resp = loadmat_single(files_mat[0])

    # load the ground thruth from the learning part
    learning = load_mapping_learning(folder)
    subj_ratings = get_subjectiv_rating(folder)

    no_rating = dict(subj_valence=np.nan,
                     subj_arousal=np.nan,
                     rating1_valence=np.nan,
                     rating1_arousal=np.nan,
                     rating2_valence=np.nan,
                     rating2_arousal=np.nan)

    resp_info = {}
    for [name], _t1, [seen_before_resp], _t2, _t3, [quad_resp], _t4 in resp:
        name = name.replace('.jpg', '')
        seen_before = name in learning.index
        quad = learning.loc[name].quad if seen_before else None

        assert name not in resp_info
        oasis_rating = get_oasis_rating(name)
        subj_rating = subj_ratings.loc[name].to_dict() if name in subj_ratings.index else no_rating
        resp_info[name] = {'seen_before': seen_before,
                           'quad': quad,
                           'seen_before_resp': seen_before_map[int(seen_before_resp)],
                           'quad_resp': quad_map[int(quad_resp)],
                           } | oasis_rating | subj_rating

    assert sum([v['seen_before'] for r, v in resp_info.items() ])==96
    assert len(resp_info)==144

    df = pd.DataFrame(resp_info).transpose()
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

def load_sleep(folder, sfreq=100):
    """load the localizer of one participant

    Parameters
    ----------
    folder : TYPE
        link to participant night.
    sfreq : TYPE, optional
        sampling frequency to downsample to. The default is None.
    event_id : TYPE, optional
        after which event to load the data. The default is 2.

    Returns
    -------
    tuple
        trials , valence, arousal
    """
    # search for files that contain "sleep" and not "test" in them
    files_vhdr = ospath.list_files(folder, patterns='*sleep*.vhdr')
    files_vhdr += ospath.list_files(folder, patterns='*Sleep*.vhdr')
    files_vhdr = [f for f in files_vhdr if not 'test' in f.lower()]
    # there should be exactly one such file
    assert len(set(files_vhdr))==1, f'too many or too few sleep vhdr files: {files_vhdr=}'

    # load the raw data into memory, preprocess and extract events
    raw = load_raw_and_preprocess(files_vhdr[0], sfreq=sfreq)
    return raw


def load_localizer(folder, sfreq=100, event_id=2, tmin=-0.5, tmax=1.5,
                     return_epochs=False):
    """load the localizer of one participant

    Parameters
    ----------
    folder : TYPE
        link to participant night.
    sfreq : TYPE, optional
        sampling frequency to downsample to. The default is None.
    event_id : TYPE, optional
        after which event to load the data. The default is 2.

    Returns
    -------
    tuple
        trials , valence, arousal
    """
    files_vhdr = ospath.list_files(folder, patterns='*localizer*.vhdr')
    assert len(files_vhdr)==1, f'too many or too few localizer vhdr files: {files_vhdr=}'

    # load the raw data into memory and extract events
    raw = load_raw_and_preprocess(files_vhdr[0], sfreq=sfreq)
    events, events_dict = mne.events_from_annotations(raw, verbose='WARNING')

    localizer_resp = load_localizer_responses(folder)
    valence_subj = localizer_resp['valence'].values
    arousal_subj = localizer_resp['arousal'].values

    attrs = [get_oasis_rating(img_name) for img_name in localizer_resp.index]
    valence_mean = [a['valence_mean'] for a in attrs]
    arousal_mean = [a['arousal_mean'] for a in attrs]
    img_category = [a['img_category'] for a in attrs]

    img_category = [settings.image_categories[cat] for cat in img_category]

    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        preload=True, verbose='WARNING')
    events_hash = hash_array(epochs.events[:,0])

    fdescriptor = f'{settings.cache_dir}/{get_file_descriptor(raw.filenames[0])}'
    epochs_file = f'{fdescriptor}-{tmin}-{tmax}-{events_hash}-epochs.pkl.gz'
    epochs, idx_removed = repair_epochs_autoreject(epochs, epochs_file)
    data_x = epochs.get_data()

    data_y = {'valence_subj': valence_subj,
              'arousal_subj': arousal_subj,
              'valence_mean': valence_mean,
              'arousal_mean': arousal_mean,
              'img_category': img_category}

    data_y = {k: np.array([v for i, v in enumerate(data_y[k]) if i not in idx_removed])
              for k in data_y}

    # valence_subj = np.array([v for i,v in enumerate(valence_subj) if i not in idx_removed])
    # arousal_subj = np.array([v for i,v in enumerate(arousal_subj) if i not in idx_removed])

    assert np.all(valence_subj<=4)
    assert np.all(valence_subj>=0)
    assert np.all(arousal_subj<=4)
    assert np.all(arousal_subj>=0)
    assert all([len(data_x)==len(y) for y in data_y.values()]), 'unequal number of trials and ratings'

    if return_epochs:
        return epochs, data_y
    return epochs.times, data_x, data_y


def load_test(folder, which, sfreq=100, event_id=2, tmin=-0.5, tmax=1.5,
              return_epochs=False):
    """
    function to load the image presentation during testing

    Parameters
    ----------
    folder : str
        link to participant night.
    which : str
        either 'before' or 'after' => Before Sleep or After Sleep test.
    sfreq : int, optional
        downsample to this frequency. The default is 100.
    event_id : TYPE, optional
        load images from this id. The default is 2.
    tmin : float, optional
        seconds before trigger onset to load. The default is -0.5.
    tmax : float, optional
        seconds after the trigger onset to load. The default is 1.5.
    return_epochs : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    np.array
        array of size (n_trials, n_channels, n_times).
    """
    assert which in ['before', 'after'], 'which must be either before or after'
    files_vhdr = ospath.list_files(folder, patterns=f'*{which}*.vhdr')
    assert len(files_vhdr)==1, f'too many or too few localizer vhdr files: {files_vhdr=}'

    # load the raw data into memory and extract events
    raw = load_raw_and_preprocess(files_vhdr[0], sfreq=sfreq)
    events, events_dict = mne.events_from_annotations(raw, verbose='WARNING')

    # load responses of this participant for this test session
    test_resp = load_test_responses(folder, which='BS' if which=='before' else 'AS')

    # cut data into epochs
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        preload=True, verbose='WARNING')



    # reject bad epochs, repair rescuable ones
    events_hash = hash_array(epochs.events[:,0]) # create unique hash on event values
    fdescriptor = f'{settings.cache_dir}/get_file_descriptor(raw.filenames[0])'
    epochs_file = f'{fdescriptor}-{tmin}-{tmax}-{events_hash}-epochs.pkl.gz'
    epochs, idx_removed = repair_epochs_autoreject(epochs, epochs_file)
    data_x = epochs.get_data()

    # convert to categorical
    img_category = [settings.image_categories[cat] for cat in test_resp.img_category]

    # put all possible target values in a dictionary
    data_y = {'seen_before_truth': test_resp.seen_before,
              'quad_truth': test_resp.quad_resp,
              'seen_before_response': test_resp.seen_before_resp,
              'quad_resp': test_resp.quad_resp,
              'valence_mean': test_resp.valence_mean,
              'arousal_mean': test_resp.arousal_mean,
              'img_category': img_category}

    data_y = {k: np.array([v for i, v in enumerate(data_y[k]) if i not in idx_removed])
              for k in data_y}

    assert all([len(data_x)==len(y) for y in data_y.values()]), 'unequal number of trials and ratings'

    if return_epochs:
        return epochs, data_y
    return epochs.times, data_x, data_y


if __name__=='__main__':
    # set some values for easier debugging
    args = ()
    kwargs={}
    vhdr_file = 'z:/Emo_React/Raw_data/PN4/Experimental night 1/PN4_Emo_React_Exp1_Localizer.vhdr'
    sfreq = 100
    event_id = 2
    tmin=-0.1
    tmax=1.5