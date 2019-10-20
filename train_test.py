
import torch
import numpy as np
from utils import all_metrics, print_metrics

def train(model, optimizer, epoch, gpu, data_loader):

    print("EPOCH %d" % epoch)

    losses = []


    model.train()

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):

        inputs_id, labels, positions, text_inputs = next(data_iter)

        inputs_id, labels, positions, text_inputs = torch.LongTensor(inputs_id), torch.FloatTensor(labels), \
                                                   torch.LongTensor(positions), text_inputs

        if gpu >= 0:
            inputs_id, labels, positions, text_inputs = inputs_id.cuda(gpu), labels.cuda(gpu), positions.cuda(gpu), text_inputs

        output, loss = model(inputs_id, labels, positions, text_inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses

def test(model, data_path, fold, gpu, dicts, data_loader):

    filename = data_path.replace('train', fold)
    print('file for evaluation: %s' % filename)
    num_labels = len(dicts['ind2c'])

    y, yhat, yhat_raw, hids, losses = [], [], [], [], []

    model.eval()

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):
        with torch.no_grad():

            inputs_id, labels, positions, text_inputs = next(data_iter)

            inputs_id, labels, positions, text_inputs = torch.LongTensor(inputs_id), torch.FloatTensor(labels), \
                                                        torch.LongTensor(positions), text_inputs

            if gpu >= 0:
                inputs_id, labels, positions, text_inputs = inputs_id.cuda(gpu), labels.cuda(gpu), positions.cuda(
                    gpu), text_inputs

            output, loss = model(inputs_id, labels, positions, text_inputs)

            output = torch.sigmoid(output)
            output = output.data.cpu().numpy()

            losses.append(loss.item())
            target_data = labels.data.cpu().numpy()

            yhat_raw.append(output)
            output = np.round(output)
            y.append(target_data)
            yhat.append(output)

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)

    k = 5 if num_labels == 50 else [8,15]
    metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
    print_metrics(metrics)
    metrics['loss_%s' % fold] = np.mean(losses)
    return metrics