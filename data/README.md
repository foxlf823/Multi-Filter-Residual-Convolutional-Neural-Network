

Our process of preparing data just follows [CAML](https://github.com/jamesmullenbach/caml-mimic) with slight modifications. 
For example, we add sentence splitting, BERT support and fasttext embedding.
Put the files of MIMIC III and II into the dir as below:
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

