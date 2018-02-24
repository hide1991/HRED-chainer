#!/usr/bin/env python

from __future__ import print_function
import argparse
import os
import random
import chainer
import numpy as np
import cupy as cp

from updater import BPTTUpdater
from chainer import training,serializers
from chainer.training import extensions
from net import Hred

def main():
    parser = argparse.ArgumentParser(description='hred')
    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')                    
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='',
                        help='Directory of image files.  Default is cifar-10.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=50,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# n_hidden: {}'.format(args.n_hidden))
    print('# epoch: {}'.format(args.epoch))
    print('')


    # Make vocab data
    # Make data [conversation number[text number[wakatigaki text]]]
    path = "dialogs"    # directry of dialogs
    vocab = {}
    id2wd = {}
    dialogs = []
    c=0
    print("loading dialog data ...")
    ### dialogファイルを再帰的にオープンして、データセットを作成する
    for (root, dirs, files) in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == u'.txt':
                with open(os.path.join(root, file),encoding='utf-8') as txt:
                    sentences = []   # init sentence
                    for line in txt:
                        line_s = line.replace(' \n', '')
                        sentences.append(line_s.split())
                        for word in line.split():
                            if word not in vocab:
                                ind = len(vocab)
                                vocab[word] = ind
                                id2wd[ind] = word
                    dialogs.append(sentences)
    # end of data load
    ind = len(vocab)
    vocab['eos'] = ind
    eos = ind
    id2wd[ind] = 'eos'

    # unknown word
    ind = len(vocab)
    vocab['unk'] = ind
    id2wd[ind] = 'unk'

    # padding word
    vocab['pad'] = -1
    id2wd[-1] = 'pad'

    # sentence reshape 45 words with padding and to ID
    # max sentence length : 43
    # max dialog length : 13
    null_sentence = []
    while True:
        null_sentence.append(-1)
        if len(null_sentence) >= 45:
            break
    dialogs_d = []
    for dialog in dialogs:
        sentences_d = []
        for sentence in dialog:
            x = []
            for word in sentence:
                x.append(vocab[word])
            x.append(vocab['eos'])
            while True:
                x.append(-1)
                # print(x)
                if len(x) >= 45:
                    break
            #print('break:',len(x))
            sentences_d.append(x)
        while True:
            sentences_d.append(null_sentence)
            if len(sentences_d) >= 15:
                #print('break:',len(sentences_d))
                break
        dialogs_d.append(sentences_d)
    #print(dialogs_d)
    #print(sentences_d)
    
    n_vocab = len(vocab)
    print("load dataset ok!","  vocab:",len(vocab))

    # Set up a neural network to train
    hred = Hred(n_vocab, args.n_hidden, args.batchsize, eos)
    
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        hred.to_gpu()  # Copy the model to the GPU
        xp = cp
    else:
        xp = np

    train = xp.array(dialogs_d, dtype=xp.int32)
    #print(xp.shape(train))
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    
    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=0.0002)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.001),'hook_dec')
        return optimizer
    optimizer = make_optimizer(hred)
    
    # Set up a trainer
    updater = BPTTUpdater(train_iter, optimizer, eos, args.bproplen, args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    
    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        hred, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    print('training start')
    trainer.run()
    #hred.test(dialogs[3][0], vocab, id2wd)

def test(model, data, vocab, id2wd):
    @training.make_extension(trigger=(5, 'epoch'))
    def _print_test(trainer):
        model.test(data[3][0], vocab, id2wd)
    return _print_test
if __name__ == '__main__':
    main()