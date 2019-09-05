from __future__ import print_function
from model import TFN
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import argparse
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import utils
import pickle # for debugging purposes

def preprocess(options):
    # parse the input args
    dataset = options['dataset']
    model_path = options['model_path']
    batch_size = options['batch_size']
    DTYPE = torch.FloatTensor
    if options['cuda']:
        DTYPE = torch.cuda.FloatTensor

    # prepare the paths for storing models
    model_path = os.path.join(
        model_path, "tfn.pt")
    print("Temp location for saving model: {}".format(model_path))

    # define fields
    text_field = 'CMU_MOSI_TimestampedWordVectors_1.1'
    visual_field = 'CMU_MOSI_VisualFacet_4.1'
    acoustic_field = 'CMU_MOSI_COVAREP'
    label_field = 'CMU_MOSI_Opinion_Labels'
    
    # DEBUG ONLY
    recalc = not (os.path.exists('vars/dump') and os.path.isfile('vars/dump'))
    
    if recalc:
        # prepare the datasets
        print("Currently using {} dataset.".format(dataset))
        DATASET = utils.download()
        dataset = utils.load(visual_field, acoustic_field, text_field)
        utils.align(text_field, dataset)
        utils.annotate(dataset, label_field)
        splits = utils.get_splits(DATASET)
        f = open('./vars/dump', 'wb+')
        pickle.dump([splits, dataset], f)
        f.close()
    else:
        f = open('./vars/dump', 'rb')
        splits, dataset = pickle.load(f)
        f.close()

    input_dims = utils.get_dims_from_dataset(dataset, text_field, acoustic_field, visual_field)
    train, dev, test = utils.split(splits, dataset, label_field, visual_field, acoustic_field, text_field, batch_size)
    train_loader, dev_loader, test_loader = utils.create_data_loader(train, dev, test, batch_size, DTYPE)
    return train_loader, dev_loader, test_loader, input_dims

def display(test_loss, test_binacc, test_precision, test_recall, test_f1, test_septacc, test_corr):
    print("MAE on test set is {}".format(test_loss))
    print("Binary accuracy on test set is {}".format(test_binacc))
    print("Precision on test set is {}".format(test_precision))
    print("Recall on test set is {}".format(test_recall))
    print("F1 score on test set is {}".format(test_f1))
    print("Seven-class accuracy on test set is {}".format(test_septacc))
    print("Correlation w.r.t human evaluation on test set is {}".format(test_corr))

def main(options):
    train_loader, valid_loader, test_loader, input_dims = preprocess(options)

    model = TFN(input_dims, (4, 16, 128), 64, (0.3, 0.3, 0.3, 0.3), 32)
    if options['cuda']:
        model = model.cuda()
    print("Model initialized")
    criterion = nn.L1Loss(size_average=False)
    optimizer = optim.Adam(list(model.parameters())[2:]) # don't optimize the first 2 params, they should be fixed (output_range and shift)
    
    # setup training
    complete = True
    min_valid_loss = float('Inf')
    patience = options['patience']
    epochs = options['epochs']
    model_path = options['model_path']
    curr_patience = patience
    for e in range(epochs):
        model.train()
        train_loss = 0.0
        num_processed = 0
        for batch in train_loader:
            num_processed += batch[0].shape[0]
            model.zero_grad()
            t, v, a, y = batch
            output = model(a, v, t)
            loss = criterion(output, y)
            loss.backward()
            train_loss += loss.data.item() / len(train_loader.dataset)
            optimizer.step()

        print("Epoch {} complete! Average Training loss: {}".format(e, train_loss))

        # Terminate the training process if run into NaN
        if np.isnan(train_loss):
            print("Training got into NaN values...\n\n")
            complete = False
            break

        # On validation set we don't have to compute metrics other than MAE and accuracy
        model.eval()
        for batch in valid_loader:
            t, v, a, y = batch
            output_valid = model(a, v, t)
            valid_loss = criterion(output, y)
        output_valid = output.cpu().data.numpy().reshape(-1)
        y = y.cpu().data.numpy().reshape(-1)

        if np.isnan(valid_loss.data[0]):
            print("Training got into NaN values...\n\n")
            complete = False
            break

        valid_binacc = accuracy_score(output_valid>=0, y>=0)

        print("Validation loss is: {}".format(valid_loss.data[0] / len(valid_loader.dataset)))
        print("Validation binary accuracy is: {}".format(valid_binacc))

        if (valid_loss.data[0] < min_valid_loss):
            curr_patience = patience
            min_valid_loss = valid_loss.data[0]
            torch.save(model, model_path)
            print("Found new best model, saving to disk...")
        else:
            curr_patience -= 1
        
        if curr_patience <= 0:
            break
        print("\n\n")

    if complete:
        
        best_model = torch.load(model_path)
        best_model.eval()
        for batch in test_loader:
            t, v, a, y = batch
            output_test = model(a, v, t)
            loss_test = criterion(output_test, y)
            test_loss = loss_test.data[0]
        output_test = output_test.cpu().data.numpy().reshape(-1)
        y = y.cpu().data.numpy().reshape(-1)

        test_binacc = accuracy_score(output_test>=0, y>=0)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y>=0, output_test>=0, average='binary')
        test_septacc = (output_test.round() == y.round()).mean()

        # compute the correlation between true and predicted scores
        test_corr = np.corrcoef([output_test, y])[0][1]  # corrcoef returns a matrix
        test_loss = test_loss / len(test_loader.dataset)

        display(test_loss, test_binacc, test_precision, test_recall, test_f1, test_septacc, test_corr)
    return

if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--dataset', dest='dataset',
                         type=str, default='MOSI')
    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=50)
    OPTIONS.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    # PATIENCE SET LOW FOR TEST PURPOSES, must increase the default back to 20
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=5)
    OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=False)
    OPTIONS.add_argument('--model_path', dest='model_path',
                         type=str, default='models')
    OPTIONS.add_argument('--max_len', dest='max_len', type=int, default=20)
    PARAMS = vars(OPTIONS.parse_args())
    main(PARAMS)
