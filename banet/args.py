# coding: utf-8

'''
这里存放一些参数
'''
import os
import time


# 训练相关的超参数
num_epochs = 100
batch_size = 100
learning_rate = 3e-4
ss_factor = 24
use_cuda = True
use_checkpoint = False
time_format = '%m-%d_%X'
current_time = time.strftime(time_format, time.localtime())
env_tag = '%s' % (current_time)
log_environment = os.path.join('logs', env_tag)  # tensorboard的记录环境


# 模型相关的超参数
projected_size = 500
hidden_size = 1024  # 循环网络的隐层单元数目
mid_size = 128  # 边界检测层的中间表示维度

frame_shape = (3, 224, 224)  # 视频帧的形状
a_feature_size = 2048  # 表观特征的大小
m_feature_size = 4096  # 运动特征的大小
feature_size = a_feature_size + m_feature_size  # 最终特征大小
#feature_size = a_feature_size  # 最终特征大小
frame_sample_rate = 10  # 视频帧的采样率
max_frames = 20  # 图像序列的最大长度
num_frames = 20  # 图像序列的最大长度
clip_num = 20
max_words = 30  # 文本序列的最大长度


# 数据相关的参数
# 提供两个数据集：MSR-VTT和MSVD
msrvtt_video_root = '/disk/private-data/yy/BDnet/BDdata/train/TrainValVideo'
msrvtt_anno_json_path = '/disk/private-data/yy/BDnet/BDdata/train/videodatainfo_2017.json'
msrvtt_video_sort_lambda = lambda x: int(x[5:-4])
msrvtt_train_range = (0, 6512)
msrvtt_val_range = (6513, 7009)
msrvtt_test_range = (7010, 9999)

msvd_video_root = './datasets/MSVD/youtube_videos'
msvd_csv_path = './datasets/MSVD/MSR Video Description Corpus_refine.csv'  # 手动修改一些数据集中的错误
msvd_video_name2id_map = './datasets/MSVD/youtube_mapping.txt'
msvd_anno_json_path = './datasets/MSVD/annotations.json'  # MSVD并未提供这个文件，需要自己写代码生成（build_msvd_annotation.py）
msvd_video_sort_lambda = lambda x: int(x[3:-4])
msvd_train_range = (0, 1200)
msvd_val_range = (1200, 1300)
msvd_test_range = (1300, 1970)


dataset = {
    'msr-vtt': [msrvtt_video_root, msrvtt_video_sort_lambda, msrvtt_anno_json_path,
                msrvtt_train_range, msrvtt_val_range, msrvtt_test_range],
    'msvd': [msvd_video_root, msvd_video_sort_lambda, msvd_anno_json_path,
             msvd_train_range, msvd_val_range, msvd_test_range]
}

# 用video_root和anno_json_path这两个变量来切换所使用的数据集
# video_sort_lambda用来对视频按照名称进行排序
#ds = 'msvd'
ds = 'msr-vtt'
video_root, video_sort_lambda, anno_json_path, \
    train_range, val_range, test_range = dataset[ds]

feat_dir = 'feats'
if not os.path.exists(feat_dir):
    os.mkdir(feat_dir)

vocab_pkl_path = os.path.join(feat_dir, ds + '_vocab.pkl')
caption_pkl_path = os.path.join(feat_dir, ds + '_captions.pkl')
caption_pkl_base = os.path.join(feat_dir, ds + '_captions')
train_caption_pkl_path = caption_pkl_base + '_train.pkl'
val_caption_pkl_path = caption_pkl_base + '_val.pkl'
test_caption_pkl_path = caption_pkl_base + '_test.pkl'

feature_h5_path = os.path.join(feat_dir, ds + '_features.h5')
feature_h5_feats = 'feats'
feature_h5_lens = 'lens'


# 结果评估相关的参数
result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

val_reference_txt_path = os.path.join(result_dir, ds + '_val_references.txt')
val_prediction_txt_path = os.path.join(result_dir, ds + '_val_predictions.txt')

test_reference_txt_path = os.path.join(result_dir, ds + '_test_references.txt')
test_prediction_txt_path = os.path.join(result_dir, ds + '_test_predictions.txt')


# checkpoint相关的超参数
resnet_checkpoint = './models/resnet50-19c8e357.pth'  # 直接用pytorch训练的模型
vgg_checkpoint = './models/vgg16-00b39a1b.pth'  # 从caffe转换而来
c3d_checkpoint = './models/c3d.pickle'

banet_pth_path = os.path.join(result_dir, ds + '_banet.pth')
best_banet_pth_path = os.path.join(result_dir, ds + '_best_banet.pth')
optimizer_pth_path = os.path.join(result_dir, ds + '_optimizer.pth')
best_optimizer_pth_path = os.path.join(result_dir, ds + '_best_optimizer.pth')


# 图示结果相关的超参数
visual_dir = 'visuals'
if not os.path.exists(visual_dir):
    os.mkdir(visual_dir)
