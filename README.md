# Multi Filter Residual Convolutional Neural Network for Text Classification

The Multi Filter Residual Convolutional Neural Network (MultiResCNN) is built based on [TextCNN](https://github.com/yoonkim/CNN_sentence), [Residual Network](https://github.com/KaimingHe/deep-residual-networks) and [CAML](https://github.com/jamesmullenbach/caml-mimic). It could be used as a strong baseline model for text classification. 

Currently, this repository only contains partial code. We will clean our code and release all of them soon. We will also provide detailed instructions on how to use them.

Main Requirements
-----
* gensim                    3.4.0
* nltk                      3.3
* numpy                     1.15.1
* python                    3.6.7
* pytorch                   1.0.1
* scikit-learn              0.20.0

Usage
-----
1. Training
  ```
  python main.py -data_path ./data/train_file -vocab ./data/vocab.csv -model MultiResCNN -n_epochs 500 -embed_file ./data/embed --gpu 0
  ```

Acknowledgement
-----
We thank all the people that provide their code to help us complete this project.