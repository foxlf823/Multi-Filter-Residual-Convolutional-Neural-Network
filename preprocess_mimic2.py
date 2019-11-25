
import codecs
from options import args
from utils import build_vocab, word_embeddings, fasttext_embeddings, gensim_to_fasttext_embeddings, gensim_to_embeddings
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from tqdm import tqdm
import csv
import re

term_pattern = re.compile('[A-Za-z]+')
with open('%s/MIMIC_RAW_DSUMS' % (args.MIMIC_2_DIR), 'r') as f:
    with open('%s/MIMIC_FILTERED_DSUMS' % (args.MIMIC_2_DIR), 'w') as f2:
        for i, line in enumerate(f):
            raw_dsum = line.split('|')[6]

            raw_dsum = re.sub(r'\[[^\]]+\]', ' ', raw_dsum)
            raw_dsum = re.sub(r'admission date:', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'discharge date:', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'date of birth:', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'sex:', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'service:', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'dictated by:.*$', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'completed by:.*$', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'signed electronically by:.*$', ' ', raw_dsum, flags=re.I)

            tokens = [token.lower() for token in re.findall(term_pattern, raw_dsum)]
            tokens = [token for token in tokens if len(token) > 1]

            # Determine if this DSUM should stay, if so, write to filtered DSUM file
            if len(tokens) > 0:
                f2.write(line)

tokenizer = RegexpTokenizer(r'\w+')

with codecs.open('%s/MIMIC_FILTERED_DSUMS' % args.MIMIC_2_DIR, 'r', encoding='latin-1') as f:
    with open('%s/proc_dsums.csv' % args.MIMIC_2_DIR, 'w') as of:
        r = csv.reader(f, delimiter='|')
        #header
        next(r)
        w = csv.writer(of)
        w.writerow(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT', 'LABELS'])
        for row in tqdm(r):
            note = row[6].replace('[NEWLINE]', '\n')
            tokens = [t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric()]
            text = ' '.join(tokens)
            codes = ';'.join(row[5].split(','))
            w.writerow([row[0], row[1], row[2], text, codes])

import nltk
nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')
with codecs.open('%s/MIMIC_FILTERED_DSUMS' % args.MIMIC_2_DIR, 'r', encoding='latin-1') as f:
    with open('%s/proc_dsums_sentsplit.csv' % args.MIMIC_2_DIR, 'w') as of:
        r = csv.reader(f, delimiter='|')
        #header
        next(r)
        w = csv.writer(of)
        w.writerow(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT', 'LABELS'])
        for row in tqdm(r):
            note = row[6].replace('[NEWLINE]', '\n')

            all_sents_inds = []
            generator = nlp_tool.span_tokenize(note)
            for t in generator:
                all_sents_inds.append(t)

            text = ""
            for ind in range(len(all_sents_inds)):
                start = all_sents_inds[ind][0]
                end = all_sents_inds[ind][1]

                sentence_txt = note[start:end]

                tokens = [t.lower() for t in tokenizer.tokenize(sentence_txt) if not t.isnumeric()]

                if ind == 0:
                    text += '[CLS] ' + ' '.join(tokens) + ' [SEP]'
                else:
                    text += ' [CLS] ' + ' '.join(tokens) + ' [SEP]'

            codes = ';'.join(row[5].split(','))
            w.writerow([row[0], row[1], row[2], text, codes])

train_ids = set()
test_ids = set()
with open('%s/training_indices.data' % args.MIMIC_2_DIR) as f:
    for row in f:
        train_ids.add(int(row.rstrip()))

with open('%s/testing_indices.data' % args.MIMIC_2_DIR) as f:
    for row in f:
        test_ids.add(int(row.rstrip()))

with open('%s/proc_dsums.csv' % args.MIMIC_2_DIR, 'r') as nf:
    with open('%s/test_dsums.csv' % args.MIMIC_2_DIR, 'w') as test_f:
        with open('%s/train_dsums.csv' % args.MIMIC_2_DIR, 'w') as train_f:
            r = csv.reader(nf, delimiter=',')
            test_w = csv.writer(test_f)
            train_w = csv.writer(train_f)
            #header
            header = next(r)
            #don't need chart time
            del(header[2])
            test_w.writerow(header)
            train_w.writerow(header)
            for i,row in enumerate(r):
                #don't need chart time
                del(row[2])
                if i in train_ids:
                    train_w.writerow(row)
                elif i in test_ids:
                    test_w.writerow(row)

with open('%s/proc_dsums_sentsplit.csv' % args.MIMIC_2_DIR, 'r') as nf:
    with open('%s/test_dsums_sentsplit.csv' % args.MIMIC_2_DIR, 'w') as test_f:
        with open('%s/train_dsums_sentsplit.csv' % args.MIMIC_2_DIR, 'w') as train_f:
            r = csv.reader(nf, delimiter=',')
            test_w = csv.writer(test_f)
            train_w = csv.writer(train_f)
            #header
            header = next(r)
            #don't need chart time
            del(header[2])
            test_w.writerow(header)
            train_w.writerow(header)
            for i,row in enumerate(r):
                #don't need chart time
                del(row[2])
                if i in train_ids:
                    train_w.writerow(row)
                elif i in test_ids:
                    test_w.writerow(row)


vfile = build_vocab(3, '%s/train_dsums.csv' % args.MIMIC_2_DIR, '%s/vocab.csv' % args.MIMIC_2_DIR)


df = pd.read_csv('%s/train_dsums_sentsplit.csv' % args.MIMIC_2_DIR)
df['length'] = df.apply(lambda row: len(row[2].split()) if not pd.isnull(row[2]) else 0, axis=1)
df = df[df['length'] > 1]
df = df.sort_values(['length'])
df.to_csv('%s/train.csv' % args.MIMIC_2_DIR, index=False)

df = pd.read_csv('%s/test_dsums_sentsplit.csv' % args.MIMIC_2_DIR)
df['length'] = df.apply(lambda row: len(row[2].split()) if not pd.isnull(row[2]) else 0, axis=1)
df = df[df['length'] > 1]
df = df.sort_values(['length'])
df.to_csv('%s/test.csv' % args.MIMIC_2_DIR, index=False)

w2v_file = word_embeddings('full', '%s/proc_dsums.csv' % args.MIMIC_2_DIR, 100, 3, 5)
gensim_to_embeddings('%s/processed_full.w2v' % args.MIMIC_2_DIR, '%s/vocab.csv' % args.MIMIC_2_DIR, None)

fasttext_file = fasttext_embeddings('full', '%s/proc_dsums.csv' % args.MIMIC_2_DIR, 100, 3, 5)
gensim_to_fasttext_embeddings('%s/processed_full.fasttext' % args.MIMIC_2_DIR, '%s/vocab.csv' % args.MIMIC_2_DIR, None)
