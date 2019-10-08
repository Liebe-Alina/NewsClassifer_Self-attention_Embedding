#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Mervin 
# time:10/8/2019


import csv
import numpy as np
import re


def clean(string):
    """
    :param string:
    :return: string.split().lower()
    Cleaning for all dataset
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)          # Clean all special character
    string = re.sub(r"\'s", " \'s", string)                         # Add a space before all 's
    string = re.sub(r"\'ve", " \'ve", string)                       # Add a space before all 've
    string = re.sub(r"\'d", " \'d", string)                         # Add a space before all 'd
    string = re.sub(r"\'ll", " \'ll", string)                       # Add a space before all 'll
    string = re.sub(r"n\'t", "n \'t", string)                       # Separate n't to n 't
    string = re.sub(r",", " , ", string)                            # Add a space before and after ','
    string = re.sub(r"!", " ! ", string)                            # Add a space before and after '!'
    string = re.sub(r"\(", " \( ", string)                          # Add a space before and after '('
    string = re.sub(r"\)", " \) ", string)                          # Add a space before and after ')'
    string = re.sub(r"\?", " \? ", string)                          # Add a space before and after '?'
    string = re.sub(r"\s{2,}", " ", string)                         # Change two or more blank characters to one space

    return string.strip().lower()


def one_hot_encoder(label_dense, num_classes):
    """
    Generate one hot encoder for labels
    :param label_dense:
    :param num_classes:
    :return: one_hot
    """
    num_labels = label_dense.shape[0]
    index_offset = np.array(num_labels) * num_classes
    one_hot = np.zeros((num_labels, num_classes))
    one_hot.flat[index_offset + label_dense.ravel() - 1] = 1
    return one_hot


def load_data(path):
    """
    Load train data and process
    :param path:
    :return: data, labels
    """
    data = []
    label = []
    with open(path, 'r') as f:
        readfile = csv.reader(f, delimiter=',', quotechar='"')
        for row in readfile:
            label.append(int(row[0]))
            txt = ""
            for s in row[1:]:
                txt = txt + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
            txt = clean(txt)
            data.append(txt)
    data = np.asarray(data)
    label = np.asarray(label)

    label_count = np.unique(label).shape[0]

    label_one_hot = one_hot_encoder(label, label_count)
    labels = label_one_hot.astype(np.uint8)

    return data, labels


def batch_iter(data, batch_size, num_epochs, shuffle = True):
    """
    Generate a batch iter for a dataset
    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return: None
    """
    data = np.array(data)
    data_size = len(data)
    batch_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            data_shuffled = data[shuffle_indices]
        else:
            data_shuffled = data
        for per_batch in range(batch_per_epoch):
            start = per_batch * batch_size
            end = min((per_batch + 1)*batch_size, data_size)
            yield data_shuffled[start:end]                          # Return a generator


if __name__ == "__main__":
    load_data("./data/train.csv")