from preprocess_ori_data import preprocess_ori_data, split_val_train, download_glove, create_vocab, prune_embedding
from preprocess.tools.generate_tsv import generate_tsv_main
from preprocess.process_video import extract_video_frame
from tsv2h5 import tsv2h5_main


def run_preprocess():    
    print('==> Preprocessing data.')
    preprocess_ori_data()
    split_val_train('../data/processed/train_qa_all.json',
                   '../data/processed', val_num=200)
    download_glove()
    create_vocab('../data/processed/train_qa.json',
                 '../data/processed/answer_set.txt',
                 '../data/processed/vocab.txt')
    prune_embedding('../data/processed/vocab.txt',
                    './data/glove.6B.300d.txt',
                    './data/glove6b_init_300d.npy')
    extract_video_frame('../data/processed/video', '../data/processed/video_imgs')
    generate_tsv_main()
    tsv2h5_main()


if __name__ == '__main__':
    run_preprocess()
