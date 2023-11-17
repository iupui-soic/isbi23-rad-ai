# isbi23-rad-ai
The analysis of our Human-AI assemblage learning study

The following folders consist of the preprocessed data for each modality (CXR, Mammography, MURA, Pneumothorax) and Phase type (Non-AI Phase, AI-Assisted Phase),
- data/Normal/output => Non AI Phase output
- data/AI/output => AI Assisted Phase output

The data preprocessing scripts that were used are located at src/dataPreprocess. All scripts in the folder are standalone scripts producing preprocessed data for each modality and phase type.
The CXR_postprocess.py which would need to be run after CXR_processing_AI.py, CXR_processing_Normal.py
The metrics_data_processing.py would need to be run after all other preprocessing scripts are run. The scripts creates the data for calculating relevant metrics.

EDA scripts include,
- src/time_detail.ipynb
- src/metrics_calculation.ipynb
- src/time_stats.ipynb
- src/accuracy_analysis.ipynb
