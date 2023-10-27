# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 00:33:03 2023

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


### CXR ###
path = r'..\..\data\AI'
annotations = pd.read_csv(path + '\\annotations\\Dataset_CXR_AI_annotations.csv')
model_outputs = pd.read_csv(path + '\\model_outputs\\Dataset_CXR_AI_model_outputs.csv')
dicom_metadata = pd.read_csv(path + '\\dicom_metadata\\Dataset_CXR_AI_dicom_metadata.csv')
ground_truth = pd.read_csv(ground_truths['CXR'])

# subset datasets
annotations = annotations[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'labelName', 'updatedAt', 'updatedById']]
model_outputs = model_outputs[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'labelId', 'probability']]
dicom_metadata = dicom_metadata[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'PatientID', 'StudyDescription']]

model_outputs['probability'] = model_outputs['probability'].apply(lambda x : eval(x)[0]['probability'])

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

model_outputs = model_outputs.merge(label_ids, on = ['labelId'], how = 'left')
# mo = model_outputs[model_outputs['SOPInstanceUID'] == '1.2.826.0.1.3680043.8.498.24945656279629736963843096152918965127']
# model_outputs = pd.pivot_table(model_outputs, index = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'], columns = ['labelName'], aggfunc='size', fill_value=0)

# Create a dataframe for labels and their probabilities separately
label_df = pd.pivot_table(model_outputs, index=['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'], columns=['labelName'], aggfunc='size', fill_value=0)
# label_df = label_df.reset_index()
prob_df = pd.pivot_table(model_outputs, index=['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'], columns=['labelName'], values = 'probability', aggfunc='mean', fill_value=0).add_suffix('_label_prob')
# prob_df = prob_df.reset_index()
# pro = prob_df[prob_df['SOPInstanceUID'] == '1.2.826.0.1.3680043.8.498.24945656279629736963843096152918965127']
# Concatenate the two dataframes on axis=1
model_outputs = pd.concat([label_df, prob_df], axis=1)

model_outputs = model_outputs.reset_index()

for col in label_ids['labelName'][~label_ids['labelName'].isin(model_outputs.columns[3:])]:
    model_outputs[col] = 0
    model_outputs[col + '_label_prob'] = 0
    
    
# merge datasets
# merge annotations , model outputs
output = annotations.merge(model_outputs.rename(columns={"labelId": "MLlabelId"}),
                  on = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'],
                  how = 'inner', suffixes=('_user_annotation', '_model_output'))

# merge with dicom metadata
output = output.merge(dicom_metadata,
              on = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'])

no_finding_idx = ground_truth[['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']].sum(axis = 1) == 0

ground_truth.loc[no_finding_idx, 'No Finding'] = 1
ground_truth['No Finding'] = ground_truth['No Finding'].fillna(0)

# merge with ground truth
output_df = ground_truth.merge(output,
                   on = ['PatientID', 'StudyDescription'])


output_df.to_csv(r"..\..\data\AI\output\Dataset_CXR_AI.csv", index = False)



