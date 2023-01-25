# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:18:18 2022

Script to combine two Brainvision recordings to one file using pybv

@author: Simon
"""
import ospath
import mne
import pybv

files = ospath.choose_files(title='Choose one or more MNE compatible files', 
                            exts=['vhdr', 'fif', 'edf'])

file_out = ospath.splitext(files[0])[0] + '_combined.vhdr'


raws = [mne.io.read_raw(file, preload=True) for file in files]
events_list = [mne.events_from_annotations(raw)[0] for raw in raws]
raw, events = mne.concatenate_raws(raws, events_list=events_list)
del raws

mne.export.export_raw(file_out, raw, fmt='brainvision')

print(f'\n\nwritten file to {file_out=}')