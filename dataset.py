from __future__ import print_function
import os
import sys
import json
import cPickle
import numpy as np
import utils
import h5py
import torch
import random
import tables
from torch.utils.data import Dataset

def get_score(occurance):
        if occurance == 1:
            return 1.0
        elif occurance == 2:
            return 1.0
        elif occurance == 3:
            return 1.0
        else:
            return 1.0

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace("s\'", " \'s")
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                if w in self.word2idx.keys():
                    tokens.append(self.word2idx[w])
                else:
                    tokens.append(len(self.idx2word))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d
    
    @classmethod
    def load_from_vocab_file(cls, path):
        with open(path) as f:
            lines = f.readlines()
        lines = [l.strip().decode('utf-8') for l in lines]
        idx2word = lines
        word2idx = {w: n_w for n_w, w in enumerate(idx2word)}
        return cls(word2idx, idx2word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class VideoQADataset(Dataset):
    def __init__(self, name, dictionary, infile, FIELDNAMES, answer_set, dataroot="../data/processed", max_length=30, v_features=None, img_id2idx=None, index=0):
        assert name in ['train', 'val', 'test']
        self.answer2id, self.answer_list, self.answer_score = self.gen_answer2id(answer_set)
        self.num_answer_candidates = len(self.answer2id)
        if name != "test":
            data_file_path = dataroot + "/" + name + "_qa_{}.json".format(index)
        else:
            data_file_path = dataroot + "/" + name + "_qa.json"
        self.data = json.load(open(data_file_path))
        print("Finish loading data!")
        train_files = []
        for i in self.data:
            train_files.append(str(i["video_id"]))
        self.v_features = v_features
        self.img_id2idx = img_id2idx
        #if '206_16' in self.img_id2idx:
        #    self.img_id2idx['206_17'] = self.img_id2idx['206_16']
        #if '201_17' in self.img_id2idx:
        #    self.img_id2idx['201_18'] = self.img_id2idx['201_17']
        #self.v_features = VideoQADataset.read_tsv(infile, FIELDNAMES, train_files)
        

        if name == "train":
            random.shuffle(self.data)
        self.dictionary = dictionary
        self.max_length = max_length
        self.selected_frames = list(range(1, 41))
        self.v_dim = 2048
        self.name = name

    def __getitem__(self, index):
        global v_features
        temp = self.data[index]
        # question tokens
        tokens = self.dictionary.tokenize(temp["question"], False)
        tokens = tokens[:self.max_length]
        if len(tokens) < self.max_length:
            padding = [self.dictionary.padding_idx] * (self.max_length - len(tokens))
            tokens = padding + tokens
        utils.assert_eq(len(tokens), self.max_length)
        tokens = np.array(tokens, dtype=np.int64)
        # answer id
        labels = np.zeros(self.num_answer_candidates)
        if self.name != "test":
            answer_id = [self.answer2id[i] for i in temp["answer"] if i in self.answer2id]
            for i in answer_id:
                labels[i] = self.answer_score[i]

        # video feature
        video_id = temp["video_id"]
        if self.name == "test":
            video_frame_id = [str(video_id) + "_" + str(i) for i in np.random.choice(self.selected_frames, 40)]
        elif self.name == "train":
            video_frame_id = [str(video_id) + "_" + str(i) for i in np.random.choice(self.selected_frames, 10)]
        else:
            video_frame_id = [str(video_id) + "_" + str(i) for i in np.random.choice(self.selected_frames, 40)]
        #print('L', [self.img_id2idx[i] for i in video_frame_id])
        #print('S', video_frame_id)
        #try:
        #video_features = [self.v_features[i]["features"] for i in video_frame_id]
        video_features = [self.v_features.root.features[self.img_id2idx[i]] if i in self.img_id2idx else self.v_features.root.features[0] for i in video_frame_id]
        video_features = np.concatenate(video_features, axis=0)
        #except:
            # key error
            #print('[E]', [self.img_id2idx[i] for i in video_frame_id])
            #exit()
            #return self.__getitem__(0)
            #video_features = np.zeros([36, 2048])
            #pass
        #print(video_features.shape)
        #video_features.dtype = 'float64'
        
        return video_features, tokens, labels.astype(np.float32)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def gen_answer2id(answer_set):
        answers = {}
        answer_score = {}
        with open(answer_set) as f:
            temp = f.readlines()
        a = 0
        for i in temp:
            answers[i[:-1].split("||")[0]] = a
            #answer_score[a] = float(get_score(int(i[:-1].split("||")[1])))
            answer_score[a] = 1.0
            a += 1
        temp = [i[:-1].split("||")[0] for i in temp]
        return answers, temp, answer_score

    @staticmethod
    def read_tsv(infile, FIELDNAMES, train_files):
        import base64
        import numpy as np
        import csv
        csv.field_size_limit(sys.maxsize)
        in_data = {}
        with open(infile, "r+b") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                item['image_id'] = item['image_id']
                if item["image_id"].split("_")[0] not in train_files:
                    continue
                item['image_h'] = int(item['image_h'])
                item['image_w'] = int(item['image_w'])
                item['num_boxes'] = int(item['num_boxes'])
                try:
                    for field in ['boxes', 'features']:
                        item[field] = np.frombuffer(base64.decodestring(item[field]),
                                                    dtype=np.float32).reshape((item['num_boxes'], -1))
                    in_data[item['image_id']] = item
                except ValueError:
                    print(item['image_id'])
        if "206_16" in in_data.keys():
            in_data["206_17"] = in_data["206_16"]
        if "201_17" in in_data.keys():
            in_data["201_18"] = in_data["201_17"]
        return in_data
    
