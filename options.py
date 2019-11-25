import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-MODEL_DIR', type=str, default='./models')
parser.add_argument('-DATA_DIR', type=str, default='./data')
parser.add_argument('-MIMIC_3_DIR', type=str, default='./data/mimic3')
parser.add_argument('-MIMIC_2_DIR', type=str, default='./data/mimic2')

parser.add_argument("-data_path", type=str, default='./data/mimic3/train_full.csv')
parser.add_argument("-vocab", type=str, default='./data/mimic3/vocab.csv')
parser.add_argument("-Y", type=str, default='full', choices=['full', '50'])
parser.add_argument("-version", type=str, choices=['mimic2', 'mimic3'], default='mimic3')
parser.add_argument("-MAX_LENGTH", type=int, default=2500)

# model
parser.add_argument("-model", type=str, choices=['CNN', 'MultiCNN', 'ResCNN', 'MultiResCNN', 'bert_seq_cls'], default='MultiResCNN')
parser.add_argument("-filter_size", type=str, default="3,5,9,15,19,25")
parser.add_argument("-num_filter_maps", type=int, default=50)
parser.add_argument("-conv_layer", type=int, default=1)
parser.add_argument("-embed_file", type=str, default='./data/mimic3/processed_full.embed')
parser.add_argument("-test_model", type=str, default=None)
parser.add_argument("-use_ext_emb", action="store_const", const=True, default=False)

# training
parser.add_argument("-n_epochs", type=int, default=500)
parser.add_argument("-dropout", type=float, default=0.2)
parser.add_argument("-patience", type=int, default=10)
parser.add_argument("-batch_size", type=int, default=16)
parser.add_argument("-lr", type=float, default=1e-4)
parser.add_argument("-weight_decay", type=float, default=0)
parser.add_argument("-criterion", type=str, default='prec_at_8', choices=['prec_at_8', 'f1_micro', 'prec_at_5'])
parser.add_argument("-gpu", type=int, default=-1, help='-1 if not use gpu, >=0 if use gpu')
parser.add_argument("-tune_wordemb", action="store_const", const=True, default=False)
parser.add_argument('-random_seed', type=int, default=1, help='0 if randomly initialize the model, other if fix the seed')

# elmo
parser.add_argument("-elmo_options_file", type=str, default='elmo_small/elmo_2x1024_128_2048cnn_1xhighway_options.json')
parser.add_argument("-elmo_weight_file", type=str, default='elmo_small/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
parser.add_argument("-elmo_tune", action="store_const", const=True, default=False)
parser.add_argument("-elmo_dropout", type=float, default=0)
parser.add_argument("-use_elmo", action="store_const", const=True, default=False)
parser.add_argument("-elmo_gamma", type=float, default=0.1)

# bert
parser.add_argument("-bert_dir", type=str)

args = parser.parse_args()
command = ' '.join(['python'] + sys.argv)
args.command = command
