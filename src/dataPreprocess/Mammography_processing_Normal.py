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
path = r'..\..\post-interview-scripts\data\Normal'
annotations = pd.read_csv(path + '\\annotations\\Dataset_Mammography_Normal_annotations.csv')
dicom_metadata = pd.read_csv(path + '\\dicom_metadata\\Dataset_Mammography_Normal_dicom_metadata.csv')
ground_truth = pd.read_csv(ground_truths['Mammo'])

# subset datasets
annotations = annotations[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'labelName', 'updatedAt', 'updatedById']]
dicom_metadata = dicom_metadata[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'PatientID', 'StudyDescription']]

label_ids = pd.DataFrame({"labelId":['L_Bme34B', 'L_dvM5Al', 'L_8JoaEB', 'L_8eZ2pB', 'L_84ZJ5l'],
                          "labelName":['Normal', 'Benign', 'Probably Benign', 'Probably Malignant', 'Malignant']})

annotations['updatedAt'] = pd.to_datetime(annotations['updatedAt'])
annotations = annotations[annotations['labelName'] != "Mask"]
annotations = annotations.drop_duplicates(subset = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'], keep = 'first')

# merge with dicom metadata
output = annotations.merge(dicom_metadata,
              on = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'])

# ground_truth = ground_truth[['Class', 'img_name_dest']]
ground_truth = ground_truth.drop('ROI File Location', axis = 1)
ground_truth = ground_truth.drop_duplicates()
ground_truth = ground_truth.rename(columns = {'Subject ID':'StudyDescription'})

ground_truth_merged = pd.DataFrame()
img_per_user = 30
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

output_df.to_csv(r"..\..\data\Normal\output\Dataset_Mammography_Normal.csv", index = False)



