import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import tables
from dataset import Dictionary, VideoQADataset
import base_model
from train import train
import utils
import torch.nn.functional as F


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/final_model')
    parser.add_argument('--batch_size', type=int, default=18)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument("--infile", type=str,
                        default="../data/processed/video_qa.h5")
    parser.add_argument("--answer_set", type=str, default="../data/processed/answer_set.txt")
    args = parser.parse_args()
    return args


def train_main(index):
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    dictionary = Dictionary.load_from_vocab_file('../data/processed/vocab.txt')

    img_id2idx_file = open('../data/processed/img_id2idx.txt', 'r')
    img_id2idx = {}
    for line in img_id2idx_file.read().split('\n'):
        sp = line.split(',')
        img_id2idx[sp[1]] = int(sp[0])

    v_features = tables.open_file(args.infile)

    train_dset = VideoQADataset(name='train',
                                dictionary=dictionary,
                                infile=args.infile,
                                FIELDNAMES=FIELDNAMES,
                                answer_set=args.answer_set,
                                v_features=v_features,
                                img_id2idx=img_id2idx,
                                index=index)
    eval_dset = VideoQADataset(name='val',
                               dictionary=dictionary,
                               infile=args.infile,
                               FIELDNAMES=FIELDNAMES,
                               answer_set=args.answer_set,
                               v_features=v_features,
                               img_id2idx=img_id2idx,
                               index=index)
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')

    model = nn.DataParallel(model).cuda()

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, 2, shuffle=False, num_workers=1)
    train(model, train_loader, eval_loader, args.epochs, args.output + str(index))

if __name__ == '__main__':
    train_main(0)
