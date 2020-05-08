import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import tables
from dataset import Dictionary, VideoQADataset
import base_model
from train import train
import utils
from torch .autograd import Variable
import datetime


def test(test_dset, index, answer_set='../data/processed/answer_set.txt'):
    num_hid = 1024
    model_name = "baseline0_newatt"

    batch_size = 8

    constructor = 'build_%s' % model_name
    model = getattr(base_model, constructor)(test_dset, num_hid).cuda()
    m = torch.load('saved_models/final_model{}/model.pth'.format(index))
    for i in m.keys():
        m[i[7:]] = m.pop(i)
    model.load_state_dict(m)
    del m
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model = nn.DataParallel(model).cuda()

    test_loader = DataLoader(test_dset, batch_size, shuffle=False, num_workers=1, drop_last=False)
    model.eval()
    pred_result = []
    for i, (v, q, a) in enumerate(test_loader):
        v = Variable(v).cuda()
        q = Variable(q).cuda()
        pred = model(v, q, None)
        x = torch.sigmoid(pred)
        pred_result.append(x.cpu().data.numpy())
    pred_result = np.concatenate(pred_result, axis=0)
    return pred_result


def test_all(answer_set='../data/processed/answer_set.txt'):
    FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
    infile = '../data/processed/video_qa.h5'

    op_name = "test"
    
    dictionary = Dictionary.load_from_vocab_file('../data/processed/vocab.txt')
    v_features = tables.open_file(infile)
    img_id2idx_file = open('../data/processed/img_id2idx.txt', 'r')
    img_id2idx = {}
    for line in img_id2idx_file.read().split('\n'):
        sp = line.split(',')
        img_id2idx[sp[1]] = int(sp[0])
    test_dset = VideoQADataset(name=op_name,
                            dictionary=dictionary,
                            infile=infile,
                            FIELDNAMES=FIELDNAMES,
                            answer_set=answer_set,
                            v_features=v_features,
                            img_id2idx=img_id2idx)

    # start test
    pred_result = []
    for i in range(10):
        print("==> Testing index: {}".format(i))
        res = test(test_dset ,i)
        pred_result.append(res)
    pred_result = np.max(pred_result, axis=0)
    answer_list = test_dset.answer_list
    result_ids = list(np.argmax(pred_result, axis=1))
    answers = [answer_list[i] for i in result_ids]
    test_data = test_dset.data
    submit = {}
    for i in range(len(test_data)):
        temp = test_data[i]
        video_name = temp["video_name"]
        question = temp["question"]
        answer = answers[i]
        answer = [answer]
        a = question + "," + "||".join(answer)
        if video_name not in submit.keys():
            submit[video_name] = [a]
        else:
            submit[video_name].append(a)
    submit_txt = []
    for i in submit.keys():
        a = i + "," + ",".join(submit[i]) + "\n"
        submit_txt.append(a)
    with open("../submit/submit_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt", "w") as f:
        f.writelines(submit_txt)
    print("finish")


if __name__=='__main__':
    test_all()


