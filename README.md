# Multi Filter Residual Convolutional Neural Network for Text Classification

The Multi Filter Residual Convolutional Neural Network (MultiResCNN) is built based on [TextCNN](https://github.com/yoonkim/CNN_sentence), [Residual Network](https://github.com/KaimingHe/deep-residual-networks) and [CAML](https://github.com/jamesmullenbach/caml-mimic).
It could be used as a strong baseline model for text classification. 
The repo can be used to reproduce the results in the [paper](https://arxiv.org/abs/1912.00862):

    @inproceedings{li2020multirescnn,  
     title={ICD Coding from Clinical Text Using Multi-Filter Residual Convolutional Neural Network},  
     author={Li, Fei and Yu, Hong},  
     booktitle={Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence},
     year={2020}  
    }

Setup
-----
This repo mainly requires the following packages.
* gensim                    3.4.0
* nltk                      3.3
* numpy                     1.15.1
* python                    3.6.7
* pytorch                   1.0.1
* scikit-learn              0.20.0
* allennlp                  0.8.4
* pytorch-pretrained-bert   0.6.2

Full packages are listed in requirements.txt.

Usage
-----
1. Prepare data

Our process of preparing data just follows [CAML](https://github.com/jamesmullenbach/caml-mimic) with slight modifications. 
Put the files of MIMIC III and II into the 'data' dir as below:
```
data
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
└───mimic2/
|   |   MIMIC_RAW_DSUMS
|   |   MIMIC_ICD9_mapping
|   |   training_indices.data
|   |   testing_indices.data
└───mimic3/
|   |   NOTEEVENTS.csv
|   |   DIAGNOSES_ICD.csv
|   |   PROCEDURES_ICD.csv
|   |   *_hadm_ids.csv (get from CAML)
```
Run ```python preprocess_mimic3.py``` and ```python preprocess_mimic2.py```.

2. Train and test using full MIMIC-III data
  ```
  python main.py -data_path ./data/mimic3/train_full.csv -vocab ./data/mimic3/vocab.csv -Y full -model MultiResCNN -embed_file ./data/mimic3/processed_full.embed -criterion prec_at_8 -gpu 0 -tune_wordemb
  ```
3. Train and test using top-50 MIMIC-III data
  ```
  python main.py -data_path ./data/mimic3/train_50.csv -vocab ./data/mimic3/vocab.csv -Y 50 -model MultiResCNN -embed_file ./data/mimic3/processed_full.embed -criterion prec_at_5 -gpu 0 -tune_wordemb
  ```
4. Train and test using full MIMIC-II data
  ```
  python main.py -data_path ./data/mimic2/train.csv -vocab ./data/mimic2/vocab.csv -Y full -version mimic2 -model MultiResCNN -embed_file ./data/mimic2/processed_full.embed -criterion prec_at_8 -gpu 0 -tune_wordemb  
  ```
5. If you want to use ELMo, add ```-use_elmo``` on the above commands.

6. Train and test using top-50 MIMIC-III data and BERT
  ```
  python main.py -data_path ./data/mimic3/train_50.csv -vocab ./data/mimic3/vocab.csv -Y 50 -model bert_seq_cls -criterion prec_at_5 -gpu 0 -MAX_LENGTH 512 -bert_dir <your bert dir>
  ```


Acknowledgement
-----
We thank all the people that provide their code to help us complete this project.