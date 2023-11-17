# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:49:11 2023

@author: rohan
"""

import pandas as pd

df_normal = pd.read_csv(r"..\..\data\Normal\output\Dataset_CXR_Normal.csv")
df_ai = pd.read_csv(r"..\..\data\AI\output\Dataset_CXR_AI.csv")

### Normal ###

# filter first 15 images only
df_normal = df_normal.sort_values(by=['updatedById', 'modality', 'end_time'])
df_normal['task_number'] = df_normal.groupby(['updatedById', 'modality']).cumcount() + 1
df_normal = df_normal[df_normal['task_number'] <= 15]

df_normal_melted = pd.melt(df_normal, id_vars = ['UserStudyDatasetID', 'dataset', 'modality', 'PatientID', 'StudyDescription',
                                                 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID',
                                                 'updatedById', 'start_time', 'end_time'],
        value_vars= ['Atelectasis_user_annotation', 'Cardiomegaly_user_annotation',
        'Effusion_user_annotation', 'Infiltration_user_annotation',
        'Mass_user_annotation', 'Nodule_user_annotation',
        'Pneumonia_user_annotation', 'Pneumothorax_user_annotation',
        'Consolidation_user_annotation', 'Edema_user_annotation',
        'Emphysema_user_annotation', 'Fibrosis_user_annotation',
        'Pleural_Thickening_user_annotation', 'Hernia_user_annotation',
        'No Finding_user_annotation',
        'Atelectasis', 'Cardiomegaly',
        'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
        'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
        'Pneumonia', 'Pneumothorax', 'No Finding'])


df_normal_melted.loc[df_normal_melted['variable'].str.contains('user_annotation'), 'type'] = 'User'
df_normal_melted.loc[~df_normal_melted['variable'].str.contains('user_annotation'), 'type'] = 'Truth'
df_normal_melted.loc[df_normal_melted['variable'].str.contains('user_annotation'), 'variable'] = df_normal_melted.loc[df_normal_melted['variable'].str.contains('user_annotation'), 'variable'].str.replace('_user_annotation', '')


# Pivot table to reshape dataframe
df_final = df_normal_melted.pivot_table(index=['UserStudyDatasetID', 'dataset', 'modality', 'PatientID',
       'StudyDescription', 'StudyInstanceUID', 'SeriesInstanceUID',
       'SOPInstanceUID', 'updatedById', 'variable'], columns='type', values='value', aggfunc='first').reset_index()

df_final["Truth"] = df_final["Truth"].replace(to_replace = {0:"FALSE"})
df_final["User"] = df_final["User"].replace(to_replace = {0:"FALSE"})
df_final.loc[df_final['Truth'] != "FALSE", 'Truth'] = df_final.loc[df_final['Truth'] != "FALSE", 'variable']
df_final.loc[df_final['User'] != "FALSE", 'User'] = df_final.loc[df_final['User'] != "FALSE", 'variable']

df_final.to_csv(r"..\..\data\Normal\cxr_postprocess\cxr_postprocess_normal.csv", index = False)

#### AI ###
df_ai_melted = pd.melt(df_ai, id_vars = ['UserStudyDatasetID', 'dataset', 'modality', 'PatientID', 'StudyDescription',
                                                 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID',
                                                 'updatedById', 'start_time', 'end_time'],
        value_vars= ['Atelectasis_user_annotation', 'Cardiomegaly_user_annotation',
        'Consolidation_user_annotation', 'Edema_user_annotation',
        'Effusion_user_annotation', 'Emphysema_user_annotation',
        'Fibrosis_user_annotation', 'Infiltration_user_annotation',
        'Mass_user_annotation', 'No Finding_user_annotation',
        'Nodule_user_annotation', 'Pleural_Thickening_user_annotation',
        'Pneumonia_user_annotation', 'Pneumothorax_user_annotation',
        'Hernia_user_annotation',
        'Atelectasis', 'Cardiomegaly', 'Effusion',
        'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening',
        'Hernia', 'No Finding',
        'Atelectasis_model_output',
        'Cardiomegaly_model_output', 'Consolidation_model_output',
        'Edema_model_output', 'Effusion_model_output', 'Emphysema_model_output',
        'Fibrosis_model_output', 'Infiltration_model_output',
        'Mass_model_output', 'No Finding_model_output', 'Nodule_model_output',
        'Pleural_Thickening_model_output', 'Pneumonia_model_output',
        'Pneumothorax_model_output', 'Hernia_model_output',
        'Atelectasis_label_prob',
        'Cardiomegaly_label_prob', 'Consolidation_label_prob',
        'Edema_label_prob', 'Effusion_label_prob', 'Emphysema_label_prob',
        'Fibrosis_label_prob', 'Infiltration_label_prob', 'Mass_label_prob',
        'No Finding_label_prob', 'Nodule_label_prob',
        'Pleural_Thickening_label_prob', 'Pneumonia_label_prob',
        'Pneumothorax_label_prob', 'Hernia_label_prob'])

df_ai_melted.loc[df_ai_melted['variable'].str.contains('user_annotation'), 'type'] = 'User'
df_ai_melted.loc[df_ai_melted['variable'].str.contains('model_output'), 'type'] = 'ML'
df_ai_melted.loc[df_ai_melted['variable'].str.contains('label_prob'), 'type'] = 'ML Prob'
df_ai_melted['type'] = df_ai_melted['type'].fillna("Truth")

df_ai_melted.loc[df_ai_melted['variable'].str.contains('user_annotation'), 'variable'] = df_ai_melted.loc[df_ai_melted['variable'].str.contains('user_annotation'), 'variable'].str.replace('_user_annotation', '')
df_ai_melted.loc[df_ai_melted['variable'].str.contains('model_output'), 'variable'] = df_ai_melted.loc[df_ai_melted['variable'].str.contains('model_output'), 'variable'].str.replace('_model_output', '')
df_ai_melted.loc[df_ai_melted['variable'].str.contains('label_prob'), 'variable'] = df_ai_melted.loc[df_ai_melted['variable'].str.contains('label_prob'), 'variable'].str.replace('_label_prob', '')

# Pivot table to reshape dataframe
df_final = df_ai_melted.pivot_table(index=['UserStudyDatasetID', 'dataset', 'modality', 'PatientID',
       'StudyDescription', 'StudyInstanceUID', 'SeriesInstanceUID',
       'SOPInstanceUID', 'updatedById', 'variable'], columns='type', values='value', aggfunc='first').reset_index()

df_final["Truth"] = df_final["Truth"].replace(to_replace = {0:"FALSE"})
df_final["User"] = df_final["User"].replace(to_replace = {0:"FALSE"})
df_final["ML"] = df_final["ML"].replace(to_replace = {0:"FALSE"})
df_final.loc[df_final['Truth'] != "FALSE", 'Truth'] = df_final.loc[df_final['Truth'] != "FALSE", 'variable']
df_final.loc[df_final['User'] != "FALSE", 'User'] = df_final.loc[df_final['User'] != "FALSE", 'variable']
df_final.loc[df_final['ML'] != "FALSE", 'ML'] = df_final.loc[df_final['ML'] != "FALSE", 'variable']

df_final.to_csv(r"..\..\data\AI\cxr_postprocess\cxr_postprocess_ai.csv", index = False)
