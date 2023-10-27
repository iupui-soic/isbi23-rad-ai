# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 10:55:33 2023

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


### Mura ###
path = r'..\..\data\Normal'
annotations = pd.read_csv(path + '\\annotations\\Dataset_MURA_Normal_annotations.csv')
dicom_metadata = pd.read_csv(path + '\\dicom_metadata\\Dataset_MURA_Normal_dicom_metadata.csv')
ground_truth = pd.read_csv(ground_truths['Mura'])

# subset datasets
annotations = annotations[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'labelName', 'updatedAt', 'updatedById']]
dicom_metadata = dicom_metadata[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'PatientID', 'StudyDescription']]

label_ids = pd.DataFrame({"labelId":['L_lqGL48', 'L_8ZJZy8', 'L_B6m4Z8', 'L_BooYLB'],
                          "labelName":['Normal', 'Fracture', 'Other Abnormality', 'Hardware']})

annotations['updatedAt'] = pd.to_datetime(annotations['updatedAt'])
annotations['start_time'] = annotations.groupby(['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'])['updatedAt'].transform('min')
annotations['end_time'] = annotations.groupby(['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'])['updatedAt'].transform('max')

# pivot data and fill in missing columns
annotations = pd.pivot_table(annotations, index = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'updatedById', 'start_time', 'end_time'], columns = ['labelName'], aggfunc='size', fill_value=0)
annotations = annotations.reset_index()
    
# merge datasets
# merge with dicom metadata
output = annotations.merge(dicom_metadata,
              on = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'])

# ground_truth = ground_truth[['Class', 'img_name_dest']]
ground_truth['img_name_dest'] = ground_truth['img_name_dest'].str[:-4]
ground_truth = ground_truth.drop_duplicates()
ground_truth = ground_truth.rename(columns = {'img_name_dest':'StudyDescription'})

ground_truth_merged = pd.DataFrame()
img_per_user = 60
lower_limit = 0
# i=1
for i in range(1, dicom_metadata.shape[0]//img_per_user + 1):
    print(i*img_per_user)
    upper_limit = i*img_per_user
    lower_limit = upper_limit - img_per_user
    dicom_metadata_temp = dicom_metadata[(dicom_metadata['PatientID'] > lower_limit) & (dicom_metadata['PatientID'] <= upper_limit)]
    ground_truth_temp = ground_truth[ground_truth['UserStudyDatasetID'] == i]
    ground_truth_temp = ground_truth_temp.rename(columns = {"img_name_dest" : "StudyDescription"})
    output_temp = ground_truth_temp.merge(dicom_metadata_temp, on = "StudyDescription")
    ground_truth_merged = pd.concat([ground_truth_merged, output_temp])

# merge with ground truth
output_df = ground_truth_merged.merge(output,
                   on = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'PatientID', 'StudyDescription'])

output_df.to_csv(r"..\..\data\Normal\output\Dataset_MURA_Normal.csv", index = False)



