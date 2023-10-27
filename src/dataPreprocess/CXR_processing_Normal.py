# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 02:22:25 2023

@author: rohan
"""

import os
import pandas as pd

DATASET_IDS = {'Dataset_MURA_Normal' : 'D_QqX2Zo',
               'Dataset_CXR_Normal' : 'D_oJ9AZE',
               'Dataset_Mammography_Normal' : 'D_olaw8Q'}

### Ground Truth ###

path = r'..\..\data\Normal\ground_truth'
CXRStudyList_Normal = path + '\\CXRStudyList.csv'
MammoStudyList_Normal = path + '\\MammoStudyList.csv'
MURAStudyList_Normal = path + '\\MURAStudyList.csv'

ground_truths = {'CXR' : CXRStudyList_Normal,
                 'Mammo' : MammoStudyList_Normal,
                 'Mura' : MURAStudyList_Normal}


### CXR ###
path = r'..\..\data\Normal'
annotations = pd.read_csv(path + '\\annotations\\Dataset_CXR_Normal_annotations.csv')
dicom_metadata = pd.read_csv(path + '\\dicom_metadata\\Dataset_CXR_Normal_dicom_metadata.csv')
ground_truth = pd.read_csv(ground_truths['CXR'])

# subset datasets
annotations = annotations[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'labelName', 'updatedAt', 'updatedById']]
dicom_metadata = dicom_metadata[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'PatientID', 'StudyDescription']]

label_ids = pd.DataFrame({"labelId":['L_lAwEv8', 'L_BOoE3B', 'L_dpyrP8', 'L_lV7a0d', 'L_83Kpk8', 'L_BknvN8', 'L_dDwKDl', 'L_lb6zql', 'L_8KDjDB', 'L_Bor0y8', 'L_8Ykepl', 'L_8xwgEd', 'L_d2Kkn8', 'L_BgPwZl', 'L_d2PVkB'],
                          "labelName":['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']})

# calculate start and end time on eache study
annotations['updatedAt'] = pd.to_datetime(annotations['updatedAt'])
annotations['start_time'] = annotations.groupby(['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'])['updatedAt'].transform('min')
annotations['end_time'] = annotations.groupby(['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'])['updatedAt'].transform('max')

# pivot data and fill in missing columns
annotations = pd.pivot_table(annotations, index = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'updatedById', 'start_time', 'end_time'], columns = ['labelName'], aggfunc='size', fill_value=0)
annotations = annotations.reset_index()

for col in label_ids['labelName'][~label_ids['labelName'].isin(annotations.columns[3:])]:
    annotations[col] = 0    
    
# merge datasets
output = annotations.merge(dicom_metadata,
              on = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']
              )

no_finding_idx = ground_truth[['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']].sum(axis = 1) == 0

ground_truth.loc[no_finding_idx, 'No Finding'] = 1
ground_truth['No Finding'] = ground_truth['No Finding'].fillna(0)

# merge with ground truth
output_df = ground_truth.merge(output,
                   on = ['PatientID'], 
                   suffixes=('_user_annotation', ''))


output_df.to_csv(r"..\..\data\Normal\output\Dataset_CXR_Normal.csv", index = False)
