# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 21:10:42 2023

@author: rohan
"""
import pandas as pd

root_path = r"..\..\data"
paths = {
    'CXR_AI' : root_path + r'\AI\cxr_postprocess\cxr_postprocess_ai.csv',
    'CXR_Normal' : root_path + r'\Normal\cxr_postprocess\cxr_postprocess_normal.csv',
    'MURA_AI' : root_path + r'\AI\output\Dataset_MURA_AI.csv',
    'MURA_Normal' : root_path + r'\Normal\output\Dataset_MURA_Normal.csv',
    'Mammography_AI' : root_path + r'\AI\output\Dataset_Mammography_AI.csv',
    'Mammography_Normal' : root_path + r'\Normal\output\Dataset_Mammography_Normal.csv',
    'Pneumothorax_AI' : root_path + r'\AI\output\Dataset_Pneumothorax_AI.csv',
    'Pneumothorax_Normal' : root_path + r'\Normal\output\Dataset_Pneumothorax_Normal.csv'
    }

# read files
cxr_ai = pd.read_csv(paths['CXR_AI'])
cxr_normal = pd.read_csv(paths['CXR_Normal'])
mura_ai = pd.read_csv(paths['MURA_AI'])
mura_normal = pd.read_csv(paths['MURA_Normal'])
mammography_ai = pd.read_csv(paths['Mammography_AI'])
mammography_normal = pd.read_csv(paths['Mammography_Normal'])
pneumothorax_ai = pd.read_csv(paths['Pneumothorax_AI'])
pneumothorax_normal = pd.read_csv(paths['Pneumothorax_Normal'])

### CXR ###
cxr_ai = cxr_ai[['modality', 'Truth', 'PatientID', 'User', 'updatedById', 'ML', 'ML Prob']]
cxr_ai = cxr_ai.rename(columns = {'Truth':'GroundTruth', 'User':'UserAnnotatedLabel', 'updatedById': 'UserID',
                         'ML':'ML_LabelName', 'ML Prob':'ML_probability'})

cxr_ai['type'] = 'Human'


cxr_normal = cxr_normal[['modality', 'Truth', 'PatientID', 'User', 'updatedById']]
cxr_normal = cxr_normal.rename(columns = {'Truth':'GroundTruth', 'User':'UserAnnotatedLabel', 'updatedById': 'UserID'})
cxr_normal['type'] = 'Human_AI'

### MURA ###
mura_ai = mura_ai[['Modality', 'Class', 'PatientID', 'labelName_user_annotation', 'updatedById', 'labelName_model_output', 'probability']]
mura_ai = mura_ai.rename(columns = {"Modality":"modality",
                          "Class":"GroundTruth",
                          "labelName_user_annotation":"UserAnnotatedLabel",
                          "updatedById":"UserID",
                          "labelName_model_output":"ML_LabelName",
                          "probability":"ML_probability"})
mura_ai['modality'] = "MURA"
mura_ai['type'] = 'Human_AI'
mura_ai['GroundTruth'] = mura_ai['GroundTruth'].replace(to_replace = {1:'Abnormal', 0:'Normal'})


mura_normal = mura_normal[['Modality', 'Class', 'PatientID', 'annotated_Class', 'updatedById', 'end_time']]
mura_normal = mura_normal.rename(columns = {"Modality":"modality",
                          "Class":"GroundTruth",
                          "annotated_Class":"UserAnnotatedLabel",
                          "updatedById":"UserID"})
mura_normal['modality'] = "MURA"
mura_normal['type'] = 'Human'
mura_normal['GroundTruth'] = mura_normal['GroundTruth'].replace(to_replace = {1:'Abnormal', 0:'Normal'})
mura_normal['UserAnnotatedLabel'] = mura_normal['UserAnnotatedLabel'].replace(to_replace = {1:'Abnormal', 0:'Normal'})

mura_normal = mura_normal.sort_values(by=['UserID', 'modality', 'type', 'end_time'])

# Generate task_number column
mura_normal['task_number'] = mura_normal.groupby(['UserID', 'modality', 'type']).cumcount() + 1
mura_normal = mura_normal[mura_normal['task_number'] <= 15]
mura_normal = mura_normal.drop(["task_number", "end_time"], axis = 1)

### Mammo ###
mammography_ai = mammography_ai[['modality', 'pathology', 'PatientID', 'labelName_user_annotation', 'updatedById', 'labelName_model_output', 'probability']]
mammography_ai = mammography_ai.rename(columns = {'pathology':'GroundTruth',
                                 'labelName_user_annotation':'UserAnnotatedLabel',
                                 'updatedById':'UserID',
                                 'labelName_model_output':'ML_LabelName',
                                 'probability':'ML_probability'})
mammography_ai['type'] = 'Human_AI'

mammography_normal = mammography_normal[['modality', 'pathology', 'PatientID', 'labelName', 'updatedById', 'updatedAt']]
mammography_normal = mammography_normal.rename(columns = {'pathology':'GroundTruth',
                                 'labelName':'UserAnnotatedLabel',
                                 'updatedById':'UserID',
                                 })
mammography_normal['type'] = 'Human'

mammography_normal = mammography_normal.sort_values(by=['UserID', 'modality', 'type', 'updatedAt'])

# Generate task_number column
mammography_normal['task_number'] = mammography_normal.groupby(['UserID', 'modality', 'type']).cumcount() + 1
mammography_normal = mammography_normal[mammography_normal['task_number'] <= 15]
mammography_normal = mammography_normal.drop(["task_number", "updatedAt"], axis = 1)

### Pneumo ###
pneumothorax_ai = pneumothorax_ai[['Dataset', 'class', 'PatientID', 'labelName_user_annotation', 'updatedById', 'labelName_model_output', 'probability']]
pneumothorax_ai = pneumothorax_ai.rename(columns = {'Dataset':'modality',
                                  'class':'GroundTruth',
                                  'labelName_user_annotation':'UserAnnotatedLabel',
                                  'updatedById':'UserID',
                                  'labelName_model_output':'ML_LabelName',
                                  'probability':'ML_probability'})
pneumothorax_ai['type'] = 'Human_AI'
pneumothorax_ai['GroundTruth'] = pneumothorax_ai['GroundTruth'].replace(to_replace = {1:'Pneumothorax', 0:'No Pneumothorax'})


pneumothorax_normal = pneumothorax_normal[['Dataset', 'class', 'PatientID', 'labelName', 'updatedById', 'updatedAt']]
pneumothorax_normal = pneumothorax_normal.rename(columns = {'Dataset':'modality',
                                  'class':'GroundTruth',
                                  'labelName':'UserAnnotatedLabel',
                                  'updatedById':'UserID'})
pneumothorax_normal['GroundTruth'] = pneumothorax_normal['GroundTruth'].replace(to_replace = {1:'Pneumothorax', 0:'No Pneumothorax'})

pneumothorax_normal['type'] = 'Human'

pneumothorax_normal = pneumothorax_normal.sort_values(by=['UserID', 'modality', 'type', 'updatedAt'])

# Generate task_number column
pneumothorax_normal['task_number'] = pneumothorax_normal.groupby(['UserID', 'modality', 'type']).cumcount() + 1
pneumothorax_normal = pneumothorax_normal[pneumothorax_normal['task_number'] <= 15]
pneumothorax_normal = pneumothorax_normal.drop(["task_number", "updatedAt"], axis = 1)


df = pd.concat([cxr_ai,
                cxr_normal,
                mura_ai,
               mura_normal,
               mammography_ai,
               mammography_normal,
               pneumothorax_ai,
               pneumothorax_normal])

file_path = root_path + r'\metrics.xlsx'
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    # Write the DataFrame to the 'dataAll' sheet
    df.to_excel(writer, sheet_name='dataAll', index=False)

