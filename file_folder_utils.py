# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 08:27:06 2022

@author: chmelikj
"""

import json
import os
from typing import Tuple
from shutil import copy2
from medpy import io
import numpy as np


def subdirs(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def nested_subfiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + nested_subfiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def prepare_nnUNet_file_structure(file_list, output_path, isTraining=True):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, 'imagesTr'))
        os.makedirs(os.path.join(output_path, 'imagesTs'))
        os.makedirs(os.path.join(output_path, 'labelsTr'))
        os.makedirs(os.path.join(output_path, 'labelsTs'))
    if isTraining:
        save_folder = 'imagesTr'
        dt = 'Tr'
    else:
        save_folder = 'imagesTs'
        dt = 'Ts'
    labels_errors = []
    for ind, file in enumerate(file_list):
        # save_image_name = os.path.basename(os.path.dirname(os.path.dirname(file))) + os.path.basename(os.path.dirname(file)).split('_')[1]
        save_image_name = (file.split(os.sep)[7] + '_' + dt + "%04d") % ind
        save_image_destination_and_name = os.path.join(output_path, save_folder, save_image_name  + '_0000.nii.gz')
        copy2(file, save_image_destination_and_name)
        print('\033[42m' + '({}/{}) Prepared image {}.'.format(ind+1, len(file_list), save_image_name + '\033[0m') )
        save_image_destination_and_name = save_image_destination_and_name.replace('images','labels').replace('_0000', '')
        

        if os.path.exists(file.replace('Data.nii','Mask.nii')):    
            image, header = io.load(file.replace('Data.nii','Mask.nii'))
            copy2(file.replace('Data.nii','Mask.nii'), save_image_destination_and_name)
            print('\033[42m' + '({}/{}) Prepared labels for {}.'.format(ind+1, len(file_list), save_image_name + '\033[0m'))
        else:
            print('\033[42m' + '({}/{})  Non-binary labels for {}.'.format(ind+1, len(file_list), save_image_name  + '\033[0m'))
    

def convert_nnUNet_segmentations_into_original_structure(input_path, output_path):
    file_list = subfiles(input_path, suffix='.nii.gz')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for ind, file in enumerate(file_list):
        patient_name = os.path.basename(file).split('_')[0]
        patient_path = os.path.join(output_path, patient_name)
        if not os.path.exists(patient_path):
            os.makedirs(patient_path)
        series_name = os.path.basename(file).split('_')[0] + '_' + os.path.basename(file).split('_')[1]
        series_path = os.path.join(patient_path, series_name)
        if not os.path.exists(series_path):
            os.makedirs(series_path)
        copy2(file, os.path.join(series_path, 'mprage_nnUNet_lesion.nii.gz'))
        print('\033[42m' + '({}/{}) Prepared image {}.'.format(ind+1, len(file_list), os.path.basename(file)) + '\033[0m')



def save_json(obj, file, indent=4, sort_keys=False): # !!! changed to sort_keys=False in order to sort correctly if more than 10 labels used - works OK for me, but maybe could cause some problems
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques

def generate_dataset_json_new(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    Modified version from nnUNet utils
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: list of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ['T1', 'T2', 'FLAIR'].
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    if len(modalities)==1:
        json_dict['tensorImageSize'] = "3D"
    else:
        json_dict['tensorImageSize'] = "4D"

    json_dict['numTraining'] = len(train_identifiers)

    if len(test_identifiers)==0:
        json_dict['numTest'] = []
    else:
        json_dict['numTest'] = len(test_identifiers)

    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file))