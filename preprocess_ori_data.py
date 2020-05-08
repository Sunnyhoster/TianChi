import json
import os
import shutil
import zipfile
import random
import pandas as pd
import urllib
import csv
from pandas import Series, DataFrame
import numpy as np


def unzip_ori_data(train_zip_path, extract_path):
    # unzip
    train_zip = zipfile.ZipFile(train_zip_path, 'r')
    for file in train_zip.namelist():
        train_zip.extract(file, extract_path)
    train_zip.close()


def preprocess_ori_data():
    dir_list = ('../data/processed',
                '../data/data_unzip',
                '../data/data_unzip/train',
                '../data/processed/video')
    for dir_path in dir_list:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    if not os.path.exists('../data/data_unzip/semifinal_video_phase2'):
        print('    ==> Unpacking origin data file.')
        unzip_ori_data('../data/VQA_round2_DatasetB_20181025.zip',
                       '../data/data_unzip')

        # generate video name -> id table
        train_video_list = [x.split('.')[0]
                            for x in os.listdir('../data/data_unzip/semifinal_video_phase2/train')]
        test_video_list = [x.split('.')[0]
                           for x in os.listdir('../data/data_unzip/semifinal_video_phase2/test')]

        video_idx_file = open('../data/processed/video_idx.txt', 'w')
        video_idx_dict = {}
        for idx, video_name in enumerate(train_video_list):
            video_idx_dict[video_name] = idx
            video_idx_file.write('{},{}\n'.format(idx, video_name))

        video_idx_dict_test = {}
        for idx, video_name in enumerate(test_video_list):
            video_idx_dict_test[video_name] = idx + len(video_idx_dict)
            video_idx_file.write('{},{}\n'.format(idx, video_name))

        print('    ==> Video files count: {}.'.format(len(video_idx_dict)))

        # move video
        print('    ==> Moving video files.')
        for video in os.listdir('../data/data_unzip/semifinal_video_phase2/train'):
            shutil.move('../data/data_unzip/semifinal_video_phase2/train/' + video, '../data/processed/video/' +
                        str(video_idx_dict[video.split('.')[0]]) + '.mp4')

        for video in os.listdir('../data/data_unzip/semifinal_video_phase2/test'):
            shutil.move('../data/data_unzip/semifinal_video_phase2/test/' + video, '../data/processed/video/' +
                        str(video_idx_dict_test[video.split('.')[0]]) + '.mp4')

    else:
        video_idx_file = open('../data/processed/video_idx.txt', 'r')
        video_idx_dict = {}
        for line in video_idx_file.read().split('\n'):
            sp = line.split(',')
            if len(sp) == 2:
                video_idx_dict[sp[1]] = int(sp[0])

        video_idx_dict_test = video_idx_dict

    # generate
    answer_set = set()
    print('    ==> Generating train&test qa json files.')
    train_list = []
    train_txt_file = open('../data/data_unzip/semifinal_video_phase2/train.txt',
                          'r')
    query_id = 0
    for line in train_txt_file.read().split('\n'):
        sp = line.strip().split(',')
        if len(sp) != 21:
            continue
        for ques_id in range(5):
            for ans in sp[ques_id * 4 + 2:(ques_id + 1) * 4 + 1]:
                answer_set.add(ans)
            query_dict = {'answer': sp[ques_id * 4 + 2:(ques_id + 1) * 4 + 1], 'question': sp[ques_id * 4 + 1],
                          'video_id': video_idx_dict[sp[0]], 'video_name': sp[0], 'id': query_id}
            train_list.append(query_dict)
            query_id += 1

    answer_set_file = open('../data/processed/answer_set.txt', 'w')
    answer_set_file.write('\n'.join(answer_set))

    json_file = open('../data/processed/train_qa_all.json', 'w')
    json.dump(train_list, json_file)

    test_list = []
    test_txt_file = open(
        '../data/data_unzip/semifinal_video_phase2/test.txt', 'r')
    query_id = 0
    for line in test_txt_file.readlines():
        sp = line.strip().split(',')
        if len(sp) != 21:
            continue
        for ques_id in range(5):
            query_dict = {'answer': '', 'question': sp[ques_id * 4 + 1],
                          'video_id': video_idx_dict_test[sp[0]], 'video_name': sp[0], 'id': query_id}
            test_list.append(query_dict)
            query_id += 1

    json_file = open('../data/processed/test_qa.json', 'w')
    json.dump(test_list, json_file)


def val_train_json(train, val_id_list, index, output_dir):
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    for idx in range(train.shape[0]):
        if train.iloc[idx]['video_id'] in val_id_list:
            val_df = val_df.append(train.iloc[idx])
        else:
            train_df = train_df.append(train.iloc[idx])

    train_df['video_id'] = train_df['video_id'].astype('int')
    train_df['id'] = train_df['id'].astype('int')
    val_df['video_id'] = val_df['video_id'].astype('int')
    val_df['id'] = val_df['id'].astype('int')

    val_df.to_json(os.path.join(output_dir, 'val_qa_{}.json'.format(index)), 'records')
    train_df.to_json(os.path.join(output_dir, 'train_qa_{}.json'.format(index)), 'records')


def split_val_train(train_qa_path, output_dir, val_num):
    print('    ==> Spliting train-validation data.')
    train = pd.read_json(train_qa_path)
    video_id_set = set()
    for video_id in train['video_id']:
        video_id_set.add(video_id)

    video_id_list = list(video_id_set)
    random.shuffle(video_id_list)

    val_size = int(len(video_id_list) / 10)
    for index in range(10):
        val_id_list = video_id_list[index*val_size:(index+1)*val_size]
        val_train_json(train, val_id_list, index, output_dir)

    # val_id_list = video_id_list[-val_num:]

    


def download_glove():
    if not os.path.exists('./data'):
        os.mkdir('./data')
    if os.path.exists('./data/glove.6B.zip'):
        print('    ==> Pretrained glove word-vectors already downloaded.')
        return
    print('    ==> Downloading pretrained glove word-vectors.')
    urllib.urlretrieve(
        'https://nlp.stanford.edu/data/glove.6B.zip', './data/glove.6B.zip')
    unzip_ori_data('./data/glove.6B.zip', './data')


def create_vocab(trainqa_path, answerset_path, vocab_path):
    """Create the 8000 vocabulary based on questions in train split.
    7999 most frequent words and 1 <UNK>.
    Args:
        trainqa_path: path to train_qa.json.
        vocab_path: vocabulary file.
    """
    vocab = dict()
    train_qa = pd.read_json(trainqa_path)
    # remove question whose answer is not in answerset
    answerset = pd.read_csv(answerset_path, header=None)[0]
    # train_qa = train_qa[train_qa['answer'].isin(answerset)]

    questions = train_qa['question'].values
    for q in questions:
        words = q.rstrip('?').split()
        for word in words:
            if len(word) >= 2:
                vocab[word] = vocab.get(word, 0) + 1
    vocab = Series(vocab)
    vocab.sort_values(ascending=False, inplace=True)
    vocab = DataFrame(vocab.iloc[0:7999])
    vocab.loc['<UNK>'] = [0]
    vocab.to_csv(vocab_path, columns=[], header=False)


def prune_embedding(vocab_path, glove_path, embedding_path):
    """Prune word embedding from pre-trained GloVe.
    For words not included in GloVe, set to average of found embeddings.
    Args:
        vocab_path: vocabulary path.
        glove_path: pre-trained GLoVe word embedding.
        embedding_path: .npy for vocabulary embedding.
    """
    # load GloVe embedding.
    glove = pd.read_csv(
        glove_path, sep=' ', quoting=csv.QUOTE_NONE, header=None)
    glove.set_index(0, inplace=True)
    # load vocabulary.
    vocab = pd.read_csv(vocab_path, header=None)[0]

    embedding = np.zeros([len(vocab), len(glove.columns)], np.float64)
    not_found = []
    for i in range(len(vocab)):
        word = vocab[i]
        if word in glove.index:
            embedding[i] = glove.loc[word]
        else:
            not_found.append(i)
    print('Not found:\n', vocab.iloc[not_found])

    embedding_avg = np.mean(embedding, 0)
    embedding[not_found] = embedding_avg

    np.save(embedding_path, embedding.astype(np.float32))




if __name__ == '__main__':
    unzip_ori_data('../data/VQADatasetB_train_part1_20180919.zip', '../data/data_unzip')
    unzip_ori_data('../data/VQADatasetB_train_part2_20180919.zip', '../data/data_unzip')
    unzip_ori_data('../data/VQADatasetB_test_20180919.zip', '../data/data_unzip')
