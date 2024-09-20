# SAML
SAML: Speaker Adaptive Mixture of LoRA Experts for End-to-End ASR [(Paper)](https://arxiv.org/abs/2406.19706)

# Environment
python: 3.9

cuda: 10.2

pytorch: 1.10.1

# Procedure
SAML method is divided into three steps: expert initialisation, pretraining, and adaptation.

## 0.Expert initialisation
Each group of experts has initialised with the LoRA parameters pretrained on a single speaker data from the train-clean-100 set.

## 1.Pretraining
For SAML pretraining, the train-clean-100 set is used which does not have any speaker overlap with the selected speakers.

## 2.Adaptation
SAML-based speaker adaptation is performed on speaker-specific data.
The specific information of speakers refers to [`whisper/data/train_clean_360_10spk`](https://github.com/qmgzhao/SAML/tree/main/whisper/data/train_clean_360_10spk).
