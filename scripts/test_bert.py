#!/usr/bin/env python3

import argparse
from transformers import BertTokenizer, BertForSequenceClassification
from os import path
from shutil import copyfile
from glob import glob
import pandas as pd
import numpy as np
import torch
import json
import re
import os
from torch.nn.functional import softmax

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument(
    '-d', '--data_dir',
    default=None,
    type=str,
    required=True,
    help='Path to folder that contains the list of all labels of the model as well as the test set (if -t is missing).',
)
parser.add_argument(
    '-m', '--model_name_or_dir',
    default=None,
    type=str,
    required=True,
    help='Path to folder of BERT model.'
)
parser.add_argument(
    '-n', '--nbest',
    default=10,
    type=int,
    required=False,
    help='Number of the max. rank to calculate the accuracy.'
)
parser.add_argument(
    '-t', '--test_set',
    nargs='+',
    default=None,
    type=str,
    required=False,
    help='Path to a different test set. If multiple -t paramaters are provided, then all data will be combined.'
)
parser.add_argument(
    '-p', '--prefix',
    default='',
    type=str,
    required=False,
    help='Specify a prefix for the test result files in the model directory.'
)
parser.add_argument(
    '--eval_all_checkpoints',
    action='store_true',
    required=False,
    help='Run the evaluation on all checkpoints.'
)
parser.add_argument(
    '--folds',
    action='store_true',
    required=False,
    help='Evaluate all subdirectories in the data directory and calculate the mean over all results.'
)
parser.add_argument(
    '--drop_duplicates',
    action='store_true',
    required=False,
    help='Drop duplicate test examples before evaluation.'
)

args = parser.parse_args()

def natural_sort(l):
    """Helper method to sort numbers human friendly."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def evaluate(model_name_or_dir, data_dir):
    """Evaluate the accuracies for a model and the test data."""

    # load all labels
    with open(path.join(data_dir, 'labels.json')) as f:
        labels = json.loads(f.read())

    # load model
    tokenizer = BertTokenizer.from_pretrained(model_name_or_dir)
    model = BertForSequenceClassification.from_pretrained(model_name_or_dir)

    if args.test_set:
        # load test examples provided with the -t param(s)
        df_test = pd.concat([
            pd.read_csv(path.join(s), sep='\t', names=['label', 'sent'], dtype={ 'label': str, 'sent': str })
            for s in args.test_set
        ]).reset_index()
    else:
        # load default test set of the data directory
        df_test = pd.read_csv(path.join(data_dir, 'test.tsv'), sep='\t', names=['label', 'sent'], dtype={ 'label': str, 'sent': str }).reset_index()

    # drop invalid lines
    df_test = df_test.dropna()

    if args.drop_duplicates:
        df_test.drop_duplicates(subset=['label', 'sent'], inplace=True)

    # normalize labels to string values
    df_test['label'] = df_test['label'].astype(str)

    # define column for rank of the correctly predicted example
    df_test['pred_pos'] = None

    # add column for the id of the label in the test set
    df_test['label_id'] = df_test.label.apply(lambda x: None if not x in labels else labels.index(x))

    # run prediction
    for row in df_test.itertuples():
        if not row.label_id >= 0:
            # set prediction position to maximum + 1 for invalid labels
            df_test.at[row.Index, 'pred_pos'] = len(labels)
            continue

        # encode sentence
        inputs = tokenizer.encode_plus(row.sent, add_special_tokens=True, return_tensors='pt')

        # calculate predictions
        preds = model(inputs['input_ids'], token_type_ids=inputs['token_type_ids'])[0]

        # sort predictions by their score
        indices = np.array(preds.sort(descending=True).indices[0])

        # calculate the prediction position for the true label
        df_test.at[row.Index, 'pred_pos'] = np.where(indices == row.label_id)[0][0]

    # calculate accuracy for the ranks
    accs = []
    for i in range(args.nbest):
        n_correct = len(df_test[df_test['pred_pos'] <= i])
        acc = n_correct / len(df_test)
        accs.append(acc)

    # save results to file
    with open(args.prefix + 'test_results.json', 'w') as f:
        f.write(json.dumps(accs))

    df_test.to_csv(args.prefix + 'test_results.csv', index=False)

    return accs

def print_accs(accs):
    print('%.2f / %.2f (2-best) / %.2f (3-best) / %.2f (5-best) / %.2f (10-best)' % (round(accs[0] * 100, 2), round(accs[1] * 100, 2), round(accs[2] * 100, 2), round(accs[4] * 100, 2), round(accs[9] * 100, 2)))

if __name__ == '__main__':
    print('=== Evaluating %s ===' % (args.model_name_or_dir))
    accs = evaluate(args.model_name_or_dir, args.data_dir)
    print_accs(accs)
