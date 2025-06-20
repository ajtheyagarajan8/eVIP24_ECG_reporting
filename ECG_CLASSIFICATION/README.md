# eVIP ECG Classification

This is an implementation of an deep learning approach for classifying ECG records from the PTB-XL dataset. 

## Environment Setup

Same as ../

## Data

Download the processed PTB-XL images from [oneDrive](https://unsw-my.sharepoint.com/:f:/g/personal/z5162987_ad_unsw_edu_au/EkQ4taa-WalDqOkG3BJymxwBMifPmKkGlx70CVkV2PZ17Q?e=fZslv2) and extract. File structure should be ./data/00000.hea for example.

Raw PTB-XL data can be downloaded from [this oneDrive](https://unsw-my.sharepoint.com/:f:/g/personal/z5162987_ad_unsw_edu_au/EsuLR8fTggRCosP5P_ByWh4BEZYROWfBO_qcTw9ijv4owQ?e=PCKpDR).

The scripts used for processing raw PTB-XL from to images has not been released yet. 

## Training ECG Classifiers

Run the cells in `pytorch_approach.ipynb` to train a ResNet101 model to perform image-based multi-label classification on 6 classes: ['atrial fibrillation','left bundle branch block','1st degree av block','premature atrial contraction','sinus rhythm','complete right bundle branch block'].

Model weights for a 2-epoch training run can be found [here](https://unsw-my.sharepoint.com/:f:/g/personal/z5162987_ad_unsw_edu_au/EmWOvKB1L_dArtipRv2Aw30Bwgomq4pvq6fT5b5nAzpNyQ?e=KEV1Ya), the filename is 'resnetPTBXL_weights.pth'.

The final cell is a starting point for future work in implementing a ViT for ECG classification.
