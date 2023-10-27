# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 10:55:33 2023

@author: rohan
"""

import os
import pandas as pd

DATASET_IDS = {'Dataset_MURA_AI' : 'D_ogmWBN',
               'Dataset_CXR_AI' : 'D_oAvq4Q',
               'Dataset_Mammography_AI' : 'D_Ejjw1E'}

### Ground Truth ###

path = r'..\..\data\AI\ground_truth'
CXRStudyList_AI = path + '\\CXRStudyList_AI.csv'
MammoStudyList_AI = path + '\\MammoStudyList_AI.csv'
MURAStudyList_AI = path + '\\MURAStudyList_AI.csv'

ground_truths = {'CXR' : CXRStudyList_AI,
                 'Mammo' : MammoStudyList_AI,
                 'Mura' : MURAStudyList_AI}


### Mura ###
path = r'..\..\data\AI'
annotations = pd.read_csv(path + '\\annotations\\Dataset_MURA_AI_annotations.csv')
model_outputs = pd.read_csv(path + '\\model_outputs\\Dataset_MURA_AI_model_outputs.csv')
dicom_metadata = pd.read_csv(path + '\\dicom_metadata\\Dataset_MURA_AI_dicom_metadata.csv')
ground_truth = pd.read_csv(ground_truths['Mura'])

# subset datasets
annotations = annotations[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'labelName', 'updatedAt', 'updatedById']]
model_outputs = model_outputs[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'labelId', 'probability']]
dicom_metadata = dicom_metadata[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'PatientID', 'StudyDescription']]

model_outputs['probability'] = model_outputs['probability'].apply(lambda x : eval(x)[0]['probability'])

label_ids = pd.DataFrame({"labelId":['L_dDLgQd', 'L_lbae3B'],
                          "labelName":['Normal', 'Abnormal']})

model_outputs = model_outputs.merge(label_ids, on = ['labelId'], how = 'left')
annotations['updatedAt'] = pd.to_datetime(annotations['updatedAt'])
    
# merge datasets
# merge annotations , model outputs
output = annotations.merge(model_outputs.rename(columns={"labelId": "MLlabelId"}),
                  on = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'],
                  how = 'inner', suffixes=('_user_annotation', '_model_output'))

# merge with dicom metadata
output = output.merge(dicom_metadata,
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
    if i == 10:
        i = 19
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

output_df.to_csv(r"..\..\data\AI\output\Dataset_MURA_AI.csv", index = False)



