import gensim.models
import numpy as np
from tqdm import tqdm
import csv
from scipy.sparse import csr_matrix
import gensim.models.word2vec as w2v
import gensim.models.fasttext as fasttext
import codecs
import re

def gensim_to_embeddings(wv_file, vocab_file, Y, outfile=None):
    model = gensim.models.Word2Vec.load(wv_file)
    wv = model.wv
    #free up memory
    del model

    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.strip()
            if line != '':
                vocab.add(line)
    ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}

    W, words = build_matrix(ind2w, wv)

    if outfile is None:
        outfile = wv_file.replace('.w2v', '.embed')

    #smash that save button
    save_embeddings(W, words, outfile)

def gensim_to_fasttext_embeddings(wv_file, vocab_file, Y, outfile=None):
    model = gensim.models.FastText.load(wv_file)
    wv = model.wv
    #free up memory
    del model

    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.strip()
            if line != '':
                vocab.add(line)
    ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}

    W, words = build_matrix(ind2w, wv)

    if outfile is None:
        outfile = wv_file.replace('.fasttext', '.fasttext.embed')

    #smash that save button
    save_embeddings(W, words, outfile)


def build_matrix(ind2w, wv):
    """
        Go through vocab in order. Find vocab word in wv.index2word, then call wv.word_vec(wv.index2word[i]).
        Put results into one big matrix.
        Note: ind2w starts at 1 (saving 0 for the pad character), but gensim word vectors starts at 0
    """
    W = np.zeros((len(ind2w)+1, len(wv.word_vec(wv.index2word[0])) ))
    words = ["**PAD**"]
    W[0][:] = np.zeros(len(wv.word_vec(wv.index2word[0])))
    for idx, word in tqdm(ind2w.items()):
        if idx >= W.shape[0]:
            break
        W[idx][:] = wv.word_vec(word)
        words.append(word)
    return W, words

def save_embeddings(W, words, outfile):
    with open(outfile, 'w') as o:
        #pad token already included
        for i in range(len(words)):
            line = [words[i]]
            line.extend([str(d) for d in W[i]])
            o.write(" ".join(line) + "\n")

def load_embeddings(embed_file):
    #also normalizes the embeddings
    W = []
    with open(embed_file) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W.append(vec)
        #UNK embedding, gaussian randomly initialized
        print("adding unk embedding")
        vec = np.random.randn(len(W[-1]))
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        W.append(vec)
    W = np.array(W)
    return W


class ProcessedIter(object):

    def __init__(self, Y, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                yield (row[3].split())


def word_embeddings(Y, notes_file, embedding_size, min_count, n_iter):
    modelname = "processed_%s.w2v" % (Y)
    sentences = ProcessedIter(Y, notes_file)

    model = w2v.Word2Vec(size=embedding_size, min_count=min_count, workers=4, iter=n_iter)
    print("building word2vec vocab on %s..." % (notes_file))

    model.build_vocab(sentences)
    print("training...")
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    out_file = '/'.join(notes_file.split('/')[:-1] + [modelname])
    print("writing embeddings to %s" % (out_file))
    model.save(out_file)
    return out_file


def fasttext_embeddings(Y, notes_file, embedding_size, min_count, n_iter):
    modelname = "processed_%s.fasttext" % (Y)
    sentences = ProcessedIter(Y, notes_file)

    model = fasttext.FastText(size=embedding_size, min_count=min_count, iter=n_iter)
    print("building fasttext vocab on %s..." % (notes_file))

    model.build_vocab(sentences)
    print("training...")
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    out_file = '/'.join(notes_file.split('/')[:-1] + [modelname])
    print("writing embeddings to %s" % (out_file))
    model.save(out_file)
    return out_file

import operator
def build_vocab(vocab_min, infile, vocab_filename):
    """
        INPUTS:
            vocab_min: how many documents a word must appear in to be kept
            infile: (training) data file to build vocabulary from
            vocab_filename: name for the file to output
    """
    with open(infile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # header
        next(reader)

        # 0. read in data
        print("reading in data...")
        # holds number of terms in each document
        note_numwords = []
        # indices where notes start
        note_inds = [0]
        # indices of discovered words
        indices = []
        # holds a bunch of ones
        data = []
        # keep track of discovered words
        vocab = {}
        # build lookup table for terms
        num2term = {}
        # preallocate array to hold number of notes each term appears in
        note_occur = np.zeros(400000, dtype=int)
        i = 0
        for row in reader:
            text = row[2]
            numwords = 0
            for term in text.split():
                # put term in vocab if it's not there. else, get the index
                index = vocab.setdefault(term, len(vocab))
                indices.append(index)
                num2term[index] = term
                data.append(1)
                numwords += 1
            # record where the next note starts
            note_inds.append(len(indices))
            indset = set(indices[note_inds[-2]:note_inds[-1]])
            # go thru all the word indices you just added, and add to the note occurrence count for each of them
            for ind in indset:
                note_occur[ind] += 1
            note_numwords.append(numwords)
            i += 1
        # clip trailing zeros
        note_occur = note_occur[note_occur > 0]

        # turn vocab into a list so indexing doesn't get fd up when we drop rows
        vocab_list = np.array([word for word, ind in sorted(vocab.items(), key=operator.itemgetter(1))])

        # 1. create sparse document matrix
        C = csr_matrix((data, indices, note_inds), dtype=int).transpose()
        # also need the numwords array to be a sparse matrix
        note_numwords = csr_matrix(1. / np.array(note_numwords))

        # 2. remove rows with less than 3 total occurrences
        print("removing rare terms")
        # inds holds indices of rows corresponding to terms that occur in < 3 documents
        inds = np.nonzero(note_occur >= vocab_min)[0]
        print(str(len(inds)) + " terms qualify out of " + str(C.shape[0]) + " total")
        # drop those rows
        C = C[inds, :]
        note_occur = note_occur[inds]
        vocab_list = vocab_list[inds]

        print("writing output")
        with open(vocab_filename, 'w') as vocab_file:
            for word in vocab_list:
                vocab_file.write(word + "\n")


def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits,
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code


import nltk
from nltk.tokenize import RegexpTokenizer
nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer = RegexpTokenizer(r'\w+')
def write_discharge_summaries(out_file, min_sentence_len, notes_file):

    print("processing notes file")
    with open(notes_file, 'r') as csvfile:
        with open(out_file, 'w') as outfile:
            print("writing to %s" % (out_file))
            outfile.write(','.join(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']) + '\n')
            notereader = csv.reader(csvfile)
            next(notereader)

            for line in tqdm(notereader):
                subj = int(line[1])
                category = line[6]
                if category == "Discharge summary":
                    note = line[10]

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

                    text = '"' + text + '"'
                    outfile.write(','.join([line[1], line[2], line[4], text]) + '\n')


    return out_file


def concat_data(labelsfile, notes_file, outfilename):
    """
        INPUTS:
            labelsfile: sorted by hadm id, contains one label per line
            notes_file: sorted by hadm id, contains one note per line
    """
    with open(labelsfile, 'r') as lf:
        print("CONCATENATING")
        with open(notes_file, 'r') as notesfile:

            with open(outfilename, 'w') as outfile:
                w = csv.writer(outfile)
                w.writerow(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS'])

                labels_gen = next_labels(lf)
                notes_gen = next_notes(notesfile)

                for i, (subj_id, text, hadm_id) in enumerate(notes_gen):
                    if i % 10000 == 0:
                        print(str(i) + " done")
                    cur_subj, cur_labels, cur_hadm = next(labels_gen)

                    if cur_hadm == hadm_id:
                        w.writerow([subj_id, str(hadm_id), text, ';'.join(cur_labels)])
                    else:
                        print("couldn't find matching hadm_id. data is probably not sorted correctly")
                        break

    return outfilename

def split_data(labeledfile, base_name, mimic_dir):
    print("SPLITTING")
    #create and write headers for train, dev, test
    train_name = '%s_train_split.csv' % (base_name)
    dev_name = '%s_dev_split.csv' % (base_name)
    test_name = '%s_test_split.csv' % (base_name)
    train_file = open(train_name, 'w')
    dev_file = open(dev_name, 'w')
    test_file = open(test_name, 'w')
    train_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")
    dev_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")
    test_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")

    hadm_ids = {}

    #read in train, dev, test splits
    for splt in ['train', 'dev', 'test']:
        hadm_ids[splt] = set()
        with open('%s/%s_full_hadm_ids.csv' % (mimic_dir, splt), 'r') as f:
            for line in f:
                hadm_ids[splt].add(line.rstrip())

    with open(labeledfile, 'r') as lf:
        reader = csv.reader(lf)
        next(reader)
        i = 0
        cur_hadm = 0
        for row in reader:
            #filter text, write to file according to train/dev/test split
            if i % 10000 == 0:
                print(str(i) + " read")

            hadm_id = row[1]

            if hadm_id in hadm_ids['train']:
                train_file.write(','.join(row) + "\n")
            elif hadm_id in hadm_ids['dev']:
                dev_file.write(','.join(row) + "\n")
            elif hadm_id in hadm_ids['test']:
                test_file.write(','.join(row) + "\n")

            i += 1

        train_file.close()
        dev_file.close()
        test_file.close()
    return train_name, dev_name, test_name


def next_labels(labelsfile):
    """
        Generator for label sets from the label file
    """
    labels_reader = csv.reader(labelsfile)
    # header
    next(labels_reader)

    first_label_line = next(labels_reader)

    cur_subj = int(first_label_line[0])
    cur_hadm = int(first_label_line[1])
    cur_labels = [first_label_line[2]]

    for row in labels_reader:
        subj_id = int(row[0])
        hadm_id = int(row[1])
        code = row[2]
        # keep reading until you hit a new hadm id
        if hadm_id != cur_hadm or subj_id != cur_subj:
            yield cur_subj, cur_labels, cur_hadm
            cur_labels = [code]
            cur_subj = subj_id
            cur_hadm = hadm_id
        else:
            # add to the labels and move on
            cur_labels.append(code)
    yield cur_subj, cur_labels, cur_hadm


def next_notes(notesfile):
    """
        Generator for notes from the notes file
        This will also concatenate discharge summaries and their addenda, which have the same subject and hadm id
    """
    nr = csv.reader(notesfile)
    # header
    next(nr)

    first_note = next(nr)

    cur_subj = int(first_note[0])
    cur_hadm = int(first_note[1])
    cur_text = first_note[3]

    for row in nr:
        subj_id = int(row[0])
        hadm_id = int(row[1])
        text = row[3]
        # keep reading until you hit a new hadm id
        if hadm_id != cur_hadm or subj_id != cur_subj:
            yield cur_subj, cur_text, cur_hadm
            cur_text = text
            cur_subj = subj_id
            cur_hadm = hadm_id
        else:
            # concatenate to the discharge summary and move on
            cur_text += " " + text
    yield cur_subj, cur_text, cur_hadm

def load_vocab_dict(args, vocab_file):
    vocab = set()

    with open(vocab_file, 'r') as vocabfile:
        for i, line in enumerate(vocabfile):
            line = line.rstrip()
            # if line.strip() in vocab:
            #     print(line)
            if line != '':
                vocab.add(line.strip())

    ind2w = {i + 1: w for i, w in enumerate(sorted(vocab))}
    w2ind = {w: i for i, w in ind2w.items()}

    return ind2w, w2ind

from collections import defaultdict

def load_full_codes(train_path, mimic2_dir, version='mimic3'):

    if version == 'mimic2':
        ind2c = defaultdict(str)
        codes = set()
        with open(mimic2_dir, 'r') as f:
            r = csv.reader(f)
            #header
            next(r)
            for row in r:
                codes.update(set(row[-1].split(';')))
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
    else:
        codes = set()
        for split in ['train', 'dev', 'test']:
            with open(train_path.replace('train', split), 'r') as f:
                lr = csv.reader(f)
                next(lr)
                for row in lr:
                    for code in row[3].split(';'):
                        codes.add(code)
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
    return ind2c

def load_lookups(args):

    ind2w, w2ind = load_vocab_dict(args, args.vocab)

    #get code and description lookups
    if args.Y == 'full':
        ind2c = load_full_codes(args.data_path, '%s/proc_dsums.csv' % args.MIMIC_2_DIR, version=args.version)
    else:
        codes = set()
        with open("%s/TOP_%s_CODES.csv" % (args.MIMIC_3_DIR, str(args.Y)), 'r') as labelfile:
            lr = csv.reader(labelfile)
            for i,row in enumerate(lr):
                codes.add(row[0])
        ind2c = {i:c for i,c in enumerate(sorted(codes))}

    c2ind = {c:i for i,c in ind2c.items()}

    dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2c, 'c2ind': c2ind}

    return dicts

def prepare_instance(dicts, filename, args, max_length):
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    instances = []
    num_labels = len(dicts['ind2c'])

    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        #header
        next(r)

        for row in r:

            text = row[2]

            labels_idx = np.zeros(num_labels)
            labelled = False

            for l in row[3].split(';'):
                if l in c2ind.keys():
                    code = int(c2ind[l])
                    labels_idx[code] = 1
                    labelled = True
            if not labelled:
                continue

            tokens_ = text.split()
            tokens = []
            tokens_id = []
            for token in tokens_:
                if token == '[CLS]' or token == '[SEP]':
                    continue
                tokens.append(token)
                token_id = w2ind[token] if token in w2ind else len(w2ind) + 1
                tokens_id.append(token_id)

            if len(tokens) > max_length:
                tokens = tokens[:max_length]
                tokens_id = tokens_id[:max_length]

            dict_instance = {'label': labels_idx,
                                 'tokens': tokens,
                                 "tokens_id": tokens_id}

            instances.append(dict_instance)


    return instances

from pytorch_pretrained_bert import BertTokenizer
def prepare_instance_bert(dicts, filename, args, max_length):
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    instances = []
    num_labels = len(dicts['ind2c'])

    wp_tokenizer = BertTokenizer.from_pretrained(args.bert_dir, do_lower_case=True)

    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        #header
        next(r)

        for row in r:

            text = row[2]

            labels_idx = np.zeros(num_labels)
            labelled = False

            for l in row[3].split(';'):
                if l in c2ind.keys():
                    code = int(c2ind[l])
                    labels_idx[code] = 1
                    labelled = True
            if not labelled:
                continue

            tokens_ = text.split()
            tokens = []
            for token in tokens_:
                if token == '[CLS]' or token == '[SEP]':
                    continue
                wps = wp_tokenizer.tokenize(token)
                tokens.extend(wps)

            tokens_max_len = max_length-2 # for CLS SEP
            if len(tokens) > tokens_max_len:
                tokens = tokens[:tokens_max_len]

            tokens.insert(0, '[CLS]')
            tokens.append('[SEP]')

            tokens_id = wp_tokenizer.convert_tokens_to_ids(tokens)
            masks = [1] * len(tokens)
            segments = [0] * len(tokens)

            dict_instance = {'label':labels_idx, 'tokens':tokens,
                             "tokens_id":tokens_id, "segments":segments, "masks":masks}

            instances.append(dict_instance)

    return instances

from torch.utils.data import Dataset
class MyDataset(Dataset):

    def __init__(self, X):
        self.X = X


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

def pad_sequence(x, max_len, type=np.int):

    padded_x = np.zeros((len(x), max_len), dtype=type)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    return padded_x

from elmo.elmo import batch_to_ids
def my_collate(x):

    words = [x_['tokens_id'] for x_ in x]

    seq_len = [len(w) for w in words]
    max_seq_len = max(seq_len)

    inputs_id = pad_sequence(words, max_seq_len)

    labels = [x_['label'] for x_ in x]

    text_inputs = [x_['tokens'] for x_ in x]
    text_inputs = batch_to_ids(text_inputs)

    return inputs_id, labels, text_inputs

def my_collate_bert(x):

    words = [x_['tokens_id'] for x_ in x]
    segments = [x_['segments'] for x_ in x]
    masks = [x_['masks'] for x_ in x]

    seq_len = [len(w) for w in words]
    max_seq_len = max(seq_len)

    inputs_id = pad_sequence(words, max_seq_len)
    segments = pad_sequence(segments, max_seq_len)
    masks = pad_sequence(masks, max_seq_len)

    labels = [x_['label'] for x_ in x]

    return inputs_id, segments, masks, labels


def early_stop(metrics_hist, criterion, patience):
    if not np.all(np.isnan(metrics_hist[criterion])):
        if len(metrics_hist[criterion]) >= patience:
            if criterion == 'loss_dev':
                return np.nanargmin(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
            else:
                return np.nanargmax(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
    else:
        return False

import json
def save_metrics(metrics_hist_all, model_dir):
    with open(model_dir + "/metrics.json", 'w') as metrics_file:
        #concatenate dev, train metrics into one dict
        data = metrics_hist_all[0].copy()
        data.update({"%s_te" % (name):val for (name,val) in metrics_hist_all[1].items()})
        data.update({"%s_tr" % (name):val for (name,val) in metrics_hist_all[2].items()})
        json.dump(data, metrics_file, indent=1)


import torch
def save_everything(args, metrics_hist_all, model, model_dir, params, criterion, evaluate=False):

    save_metrics(metrics_hist_all, model_dir)

    if not evaluate:
        #save the model with the best criterion metric
        if not np.all(np.isnan(metrics_hist_all[0][criterion])):
            if criterion == 'loss_dev':
                eval_val = np.nanargmin(metrics_hist_all[0][criterion])
            else:
                eval_val = np.nanargmax(metrics_hist_all[0][criterion])

            if eval_val == len(metrics_hist_all[0][criterion]) - 1:
                sd = model.cpu().state_dict()
                torch.save(sd, model_dir + "/model_best_%s.pth" % criterion)
                if args.gpu >= 0:
                    model.cuda(args.gpu)
    print("saved metrics, params, model to directory %s\n" % (model_dir))



def print_metrics(metrics):
    print()
    if "auc_macro" in metrics.keys():
        print("[MACRO] accuracy, precision, recall, f-measure, AUC")
        print("%.4f, %.4f, %.4f, %.4f, %.4f" % (metrics["acc_macro"], metrics["prec_macro"], metrics["rec_macro"], metrics["f1_macro"], metrics["auc_macro"]))
    else:
        print("[MACRO] accuracy, precision, recall, f-measure")
        print("%.4f, %.4f, %.4f, %.4f" % (metrics["acc_macro"], metrics["prec_macro"], metrics["rec_macro"], metrics["f1_macro"]))

    if "auc_micro" in metrics.keys():
        print("[MICRO] accuracy, precision, recall, f-measure, AUC")
        print("%.4f, %.4f, %.4f, %.4f, %.4f" % (metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"], metrics["auc_micro"]))
    else:
        print("[MICRO] accuracy, precision, recall, f-measure")
        print("%.4f, %.4f, %.4f, %.4f" % (metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"]))
    for metric, val in metrics.items():
        if metric.find("rec_at") != -1:
            print("%s: %.4f" % (metric, val))
    print()

def union_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)

def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)

def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1


def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)

def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)

def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)

def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)

def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic, ymic)

from sklearn.metrics import roc_curve, auc
def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    #get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        #only if there are true positives for this label
        if y[:,i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y[:,i], yhat_raw[:,i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score):
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    #macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])
    roc_auc['auc_macro'] = np.mean(aucs)

    #micro-AUC: just look at each individual prediction
    yhatmic = yhat_raw.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(ymic, yhatmic)
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc

def recall_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        denom = y[i,:].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)

def precision_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i,tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)

def all_metrics(yhat, y, k=8, yhat_raw=None, calc_auc=True):
    """
        Inputs:
            yhat: binary predictions matrix
            y: binary ground truth matrix
            k: for @k metrics
            yhat_raw: prediction scores matrix (floats)
        Outputs:
            dict holding relevant metrics
    """
    names = ["acc", "prec", "rec", "f1"]

    #macro
    macro = all_macro(yhat, y)

    #micro
    ymic = y.ravel()
    yhatmic = yhat.ravel()
    micro = all_micro(yhatmic, ymic)

    metrics = {names[i] + "_macro": macro[i] for i in range(len(macro))}
    metrics.update({names[i] + "_micro": micro[i] for i in range(len(micro))})

    #AUC and @k
    if yhat_raw is not None and calc_auc:
        #allow k to be passed as int or list
        if type(k) != list:
            k = [k]
        for k_i in k:
            rec_at_k = recall_at_k(yhat_raw, y, k_i)
            metrics['rec_at_%d' % k_i] = rec_at_k
            prec_at_k = precision_at_k(yhat_raw, y, k_i)
            metrics['prec_at_%d' % k_i] = prec_at_k
            metrics['f1_at_%d' % k_i] = 2*(prec_at_k*rec_at_k)/(prec_at_k+rec_at_k)

        roc_auc = auc_metrics(yhat_raw, y, ymic)
        metrics.update(roc_auc)

    return metrics


def _readString(f, code):
    # s = unicode()
    s = str()
    c = f.read(1)
    value = ord(c)

    while value != 10 and value != 32:
        if 0x00 < value < 0xbf:
            continue_to_read = 0
        elif 0xC0 < value < 0xDF:
            continue_to_read = 1
        elif 0xE0 < value < 0xEF:
            continue_to_read = 2
        elif 0xF0 < value < 0xF4:
            continue_to_read = 3
        else:
            raise RuntimeError("not valid utf-8 code")

        i = 0
        # temp = str()
        # temp = temp + c

        temp = bytes()
        temp = temp + c

        while i<continue_to_read:
            temp = temp + f.read(1)
            i += 1

        temp = temp.decode(code)
        s = s + temp

        c = f.read(1)
        value = ord(c)

    return s

import struct
def _readFloat(f):
    bytes4 = f.read(4)
    f_num = struct.unpack('f', bytes4)[0]
    return f_num

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()

    # emb_debug = []
    if embedding_path.find('.bin') != -1:
        with open(embedding_path, 'rb') as f:
            wordTotal = int(_readString(f, 'utf-8'))
            embedd_dim = int(_readString(f, 'utf-8'))

            for i in range(wordTotal):
                word = _readString(f, 'utf-8')
                # emb_debug.append(word)

                word_vector = []
                for j in range(embedd_dim):
                    word_vector.append(_readFloat(f))
                word_vector = np.array(word_vector, np.float)

                f.read(1)  # a line break

                embedd_dict[word] = word_vector

    else:
        with codecs.open(embedding_path, 'r', 'UTF-8') as file:
            for line in file:
                # logging.info(line)
                line = line.strip()
                if len(line) == 0:
                    continue
                # tokens = line.split()
                tokens = re.split(r"\s+", line)
                if len(tokens) == 2:
                    continue # it's a head
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    # assert (embedd_dim + 1 == len(tokens))
                    if embedd_dim + 1 != len(tokens):
                        continue
                embedd = np.zeros([1, embedd_dim])
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd


    return embedd_dict, embedd_dim

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def build_pretrain_embedding(embedding_path, word_alphabet, norm):

    embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)

    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.zeros([len(word_alphabet)+2, embedd_dim], dtype=np.float32)  # add UNK (last) and PAD (0)
    perfect_match = 0
    case_match = 0
    digits_replaced_with_zeros_found = 0
    lowercase_and_digits_replaced_with_zeros_found = 0
    not_match = 0
    for word, index in word_alphabet.items():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1

        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1

        elif re.sub('\d', '0', word) in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[re.sub('\d', '0', word)])
            else:
                pretrain_emb[index,:] = embedd_dict[re.sub('\d', '0', word)]
            digits_replaced_with_zeros_found += 1

        elif re.sub('\d', '0', word.lower()) in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[re.sub('\d', '0', word.lower())])
            else:
                pretrain_emb[index,:] = embedd_dict[re.sub('\d', '0', word.lower())]
            lowercase_and_digits_replaced_with_zeros_found += 1

        else:
            if norm:
                pretrain_emb[index, :] = norm2one(np.random.uniform(-scale, scale, [1, embedd_dim]))
            else:
                pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

    # initialize pad and unknown
    pretrain_emb[0, :] = np.zeros([1, embedd_dim], dtype=np.float32)
    if norm:
        pretrain_emb[-1, :] = norm2one(np.random.uniform(-scale, scale, [1, embedd_dim]))
    else:
        pretrain_emb[-1, :] = np.random.uniform(-scale, scale, [1, embedd_dim])


    print("pretrained word emb size {}".format(len(embedd_dict)))
    print("prefect match:%.2f%%, case_match:%.2f%%, dig_zero_match:%.2f%%, "
                 "case_dig_zero_match:%.2f%%, not_match:%.2f%%"
                 %(perfect_match*100.0/len(word_alphabet), case_match*100.0/len(word_alphabet), digits_replaced_with_zeros_found*100.0/len(word_alphabet),
                   lowercase_and_digits_replaced_with_zeros_found*100.0/len(word_alphabet), not_match*100.0/len(word_alphabet)))

    return pretrain_emb, embedd_dim