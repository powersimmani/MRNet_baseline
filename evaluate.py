import argparse
import matplotlib.pyplot as plt
import os,code
import numpy as np
import torch
from scipy import interp
from sklearn import metrics
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.metrics import auc as metric_auc
from torch.autograd import Variable
from sklearn.preprocessing import label_binarize
from loader import load_data
from model import MRNet

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--diagnosis', type=int, required=True)
    parser.add_argument('--gpu', action='store_true')
    return parser

def roc_function(labels, preds):
    #for multiclass
    n_classes= 3
    y_test_bin = np.asarray(labels)#label_binarize(y_test,classes=[i for i in range(n_classes)])
    y_pred_bin = np.asarray(label_binarize(np.argmax(preds, axis=1),classes=[i for i in range(n_classes)]))

    y_pred= np.argmax(preds, axis=1)
    y_test =np.argmax(labels, axis=1)
    #print(roc_auc_score(y_test, y_pred))
    #print(roc_auc_score(y_test_bin, y_pred_bin))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        #code.interact(local=dict(globals(), **locals()))
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = metric_auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])  
    
    # Compute macro-average ROC curve and ROC area
    #code.interact(local=dict(globals(), **locals()))

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metric_auc(fpr["macro"], tpr["macro"])    
    return roc_auc
    #code.interact(local=dict(globals(), **locals()))

def run_model(model, loader, train=False, optimizer=None):
    preds = []
    labels = []

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.
    num_batches = 0

    for batch in loader:
        if train:
            optimizer.zero_grad()

        vol, label = batch
        if loader.dataset.use_gpu:
            vol = vol.cuda()
            label = label.cuda()
        vol = Variable(vol)
        label = Variable(label)

        logit = model.forward(vol)

        loss = loader.dataset.weighted_loss(logit, label)
        total_loss += loss.item()

        pred = torch.sigmoid(logit)
        #pred_npy = pred.data.cpu().numpy()[0][0]#for binary classification
        pred_npy = pred.data.cpu().numpy()[0]
        label_npy = label.data.cpu().numpy()[0][0]

        #preds.append(pred_npy)#for binary classification
        preds.append(pred_npy)
        labels.append(label_npy)
        if train:
            loss.backward()
            optimizer.step()
        num_batches += 1

    avg_loss = total_loss / num_batches

    #fpr, tpr, threshold = metrics.roc_curve(labels, preds)#for binary classification
    auc = roc_function(labels, preds)#for multi classification
    
    #auc = metrics.auc(fpr, tpr)#for binary classification

    return avg_loss, auc, preds, labels

def evaluate(split, model_path, diagnosis, use_gpu):
    train_loader, valid_loader, test_loader = load_data(diagnosis, use_gpu)

    model = MRNet()
    state_dict = torch.load(model_path, map_location=(None if use_gpu else 'cpu'))
    model.load_state_dict(state_dict)

    if use_gpu:
        model = model.cuda()

    if split == 'train':
        loader = train_loader
    elif split == 'valid':
        loader = valid_loader
    elif split == 'test':
        loader = test_loader
    else:
        raise ValueError("split must be 'train', 'valid', or 'test'")

    loss, auc, preds, labels = run_model(model, loader)

    print(f'{split} loss: {loss:0.4f}')
    print(f'{split} AUC: {auc:0.4f}')

    return preds, labels

if __name__ == '__main__':
    args = get_parser().parse_args()
    evaluate(args.split, args.model_path, args.diagnosis, args.gpu)
