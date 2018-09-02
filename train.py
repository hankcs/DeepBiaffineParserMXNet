# -*- coding: UTF-8 -*-

import argparse
import os
import pickle
import sys
import time
import random
from os.path import isfile

import numpy as np
import mxnet as mx
from mxnet import autograd, gluon

from common.data import DataLoader, Vocab
from common.utils import eprint
from models import BiaffineParser
from run.config import Config
from test import test

if __name__ == "__main__":
    np.random.seed(666)
    random.seed(666)
    mx.random.seed(666)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config_file', default='configs/default.ini')
    args, extra_args = arg_parser.parse_known_args()
    if not isfile(args.config_file):
        eprint('%s not exist' % args.config_file)
        exit(1)
    config = Config(args.config_file, extra_args)

    vocab = Vocab(config.train_file, None if config.debug else config.pretrained_embeddings_file,
                  config.min_occur_count)
    if not config.debug:
        pickle.dump(vocab, open(config.save_vocab_path, 'wb'))
    with (mx.gpu(0) if 'cuda' in os.environ['PATH'] else mx.cpu()):
        parser = BiaffineParser(vocab, config.word_dims, config.tag_dims, config.dropout_emb, config.lstm_layers,
                                config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden,
                                config.mlp_arc_size,
                                config.mlp_rel_size, config.dropout_mlp, config.debug)
        data_loader = DataLoader(config.train_file, config.num_buckets_train, vocab)
        # trainer = dy.AdamTrainer(pc, config.learning_rate, config.beta_1, config.beta_2, config.epsilon)
        trainer = gluon.Trainer(parser.collect_params(), 'adam', {'learning_rate': config.learning_rate,
                                                                  'beta1': config.beta_1,
                                                                  'beta2': config.beta_2,
                                                                  'epsilon': config.epsilon
                                                                  })
        global_step = 0
        epoch = 1
        best_UAS = 0.
        history = lambda x, y: open(os.path.join(config.save_dir, 'valid_history'), 'a').write('%.2f %.2f\n' % (x, y))
        while global_step < config.train_iters:
            # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ' Start training epoch #%d' % (epoch,))
            start_time = time.time()
            for words, tags, arcs, rels in data_loader.get_batches(batch_size=config.train_batch_size, shuffle=True):
                with autograd.record():
                    arc_accuracy, rel_accuracy, overall_accuracy, loss = parser.run(words, tags, arcs,
                                                                                    rels)
                    loss = loss * 0.5
                loss.backward()
                trainer.step(config.train_batch_size)
                loss_value = loss.asscalar()  # asscalar is blocking, so it's better to delay the
                # call until backward() and step() are called
                print("\rStep #%d: Acc: arc %.2f, rel %.2f, overall %.2f, loss %.3f" % (
                    global_step, arc_accuracy, rel_accuracy, overall_accuracy, loss_value), end='')
                sys.stdout.flush()
                # trainer.set_learning_rate(config.learning_rate * config.decay ** (global_step / config.decay_steps))
                global_step += 1
                if global_step % config.validate_every == 0:
                    epoch += 1
                    print('\nduration : {:.2f}'.format(time.time() - start_time))
                    print('Test on development set')
                    LAS, UAS = test(parser, vocab, config.num_buckets_valid, config.test_batch_size, config.dev_file,
                                    os.path.join(config.save_dir, 'valid_tmp'))
                    history(LAS, UAS)
                    if global_step > config.save_after and UAS > best_UAS:
                        best_UAS = UAS
                        parser.save(config.save_model_path)
        parser.load(config.save_model_path)
        print('Test score')
        LAS, UAS = test(parser, vocab, config.num_buckets_valid, config.test_batch_size, config.dev_file,
                        os.path.join(config.save_dir, 'valid_tmp'))