
from options import args, MAX_LENGTH
import random
import numpy as np
import torch
import csv
import sys
from utils import load_lookups, make_param_dict, prepare_instance, MyDataset, my_collate, early_stop, save_everything
from models import *
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader
import os
import time
from train_test import train, test

if __name__ == "__main__":


    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    print(args)

    csv.field_size_limit(sys.maxsize)

    # load vocab and other lookups
    desc_embed = args.lmbda > 0
    print("loading lookups...")
    dicts = load_lookups(args, desc_embed=desc_embed)

    Y = len(dicts['ind2c'])
    if args.model == 'CNN':
        model = CNN(args, Y, dicts)
    elif args.model == 'MultiCNN':
        model = MultiCNN(args, Y, dicts)
    elif args.model == 'ResCNN':
        model = ResCNN(args, Y, dicts)
    elif args.model == 'MultiResCNN':
        model = MultiResCNN(args, Y, dicts)

    if args.test_model:
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    print(model)

    if not args.test_model:
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    else:
        optimizer = None

    if args.tune_wordemb == False:
        model.freeze_net()

    params = make_param_dict(args)

    metrics_hist = defaultdict(lambda: [])
    metrics_hist_te = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: [])

    test_only = args.test_model is not None
    evaluate = args.test_model is not None

    train_instances = prepare_instance(dicts, args.data_path, args, MAX_LENGTH)
    print("train_instances {}".format(len(train_instances)))
    if args.version != 'mimic2':
        dev_instances = prepare_instance(dicts, args.data_path.replace('train','dev'), args, MAX_LENGTH)
        print("dev_instances {}".format(len(dev_instances)))
    else:
        dev_instances = None
    test_instances = prepare_instance(dicts, args.data_path.replace('train','test'), args, MAX_LENGTH)
    print("test_instances {}".format(len(test_instances)))

    train_loader = DataLoader(MyDataset(train_instances), args.batch_size, shuffle=True, collate_fn=my_collate)
    if args.version != 'mimic2':
        dev_loader = DataLoader(MyDataset(dev_instances), 1, shuffle=False, collate_fn=my_collate)
    else:
        dev_loader = None
    test_loader = DataLoader(MyDataset(test_instances), 1, shuffle=False, collate_fn=my_collate)


    for epoch in range(args.n_epochs):

        if epoch == 0 and not args.test_model:
            model_dir = os.path.join(args.MODEL_DIR, '_'.join([args.model, time.strftime('%b_%d_%H:%M:%S', time.localtime())]))
            #os.mkdir(model_dir)
            os.makedirs(model_dir)
        elif args.test_model:
            model_dir = os.path.dirname(os.path.abspath(args.test_model))

        if not test_only:
            epoch_start = time.time()
            losses = train(model, optimizer, epoch, args.gpu, train_loader)
            loss = np.mean(losses)
            epoch_finish = time.time()
            print("epoch finish in %.2fs, loss: %.4f" % (epoch_finish - epoch_start, loss))
        else:
            loss = np.nan

        fold = 'test' if args.version == 'mimic2' else 'dev'
        dev_instances = test_instances if args.version == 'mimic2' else dev_instances
        dev_loader = test_loader if args.version == 'mimic2' else dev_loader
        if epoch == args.n_epochs - 1:
            print("last epoch: testing on test and train sets")
            testing = True
            quiet = False

        # test on dev
        evaluation_start = time.time()
        metrics = test(model, args.data_path, fold, args.gpu, dicts, dev_loader)
        evaluation_finish = time.time()
        print("evaluation finish in %.2fs" % (evaluation_finish - evaluation_start))
        if testing or epoch == args.n_epochs - 1:
            print("\nevaluating on test")
            metrics_te = test(model, args.data_path, "test", args.gpu, dicts, test_loader)
        else:
            metrics_te = defaultdict(float)
            fpr_te = defaultdict(lambda: [])
            tpr_te = defaultdict(lambda: [])
        metrics_tr = {'loss': loss}
        metrics_all = (metrics, metrics_te, metrics_tr)


        for name in metrics_all[0].keys():
            metrics_hist[name].append(metrics_all[0][name])
        for name in metrics_all[1].keys():
            metrics_hist_te[name].append(metrics_all[1][name])
        for name in metrics_all[2].keys():
            metrics_hist_tr[name].append(metrics_all[2][name])
        metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)

        save_everything(args, metrics_hist_all, model, model_dir, params, args.criterion, evaluate)

        sys.stdout.flush()

        if test_only:
            break

        if args.criterion in metrics_hist.keys():
            if early_stop(metrics_hist, args.criterion, args.patience):
                #stop training, do tests on test and train sets, and then stop the script
                print("%s hasn't improved in %d epochs, early stopping..." % (args.criterion, args.patience))
                test_only = True
                args.test_model = '%s/model_best_%s.pth' % (model_dir, args.criterion)
                if args.model == 'CNN':
                    model = CNN(args, Y, dicts)
                elif args.model == 'MultiCNN':
                    model = MultiCNN(args, Y, dicts)
                elif args.model == 'ResCNN':
                    model = ResCNN(args, Y, dicts)
                elif args.model == 'MultiResCNN':
                    model = MultiResCNN(args, Y, dicts)



