from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import cv2
import h5py
import numpy as np
import tensorflow as tf
import skimage
from model import AppearanceEncoder, MotionEncoder
from args import video_root, video_sort_lambda
from args import feature_h5_path, feature_h5_feats, feature_h5_lens
from args import max_frames, feature_size, clip_num
import sys
import inspect
sys.path.append("..")
from util.preprocess import VideoC3DExtractor
from util.preprocess import VideoResExtractor
import tensorflow as tf
from util.c3d import c3d
import numpy as np

#aencoder, mencoder
def sample_frames(video_path, train=True):
    '''
    对视频帧进行采样，减少计算量。等间隔地取max_frames帧
    '''
#     max_frames = 60
    frames = []
    frame_count = 0
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()

        if ret is False:
            break
        a=frame
        frame = frame[:, :, ::-1]         #???????????????
        b=frame
        frames.append(frame)
        frame_count += 1
    if frames == []:
        print("can not open %s", video_path)
    
    indices = np.linspace(8, frame_count - 7, max_frames, endpoint=False, dtype=int)

    frames = np.array(frames)
    frame_list = frames[indices]
    clip_list = []
    for index in indices:
        clip_list.append(frames[index - 8: index + 8])
    clip_list = np.array(clip_list)
    return frame_list, clip_list, frame_count


def resize_frame(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        # 把单通道的灰度图复制三遍变成三通道的图片
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, channels = image.shape
    if height == width:
        resized_image = cv2.resize(image, (target_height, target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * target_height / height),
                                           target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,
                                      cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height,
                                           int(height * target_width / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:
                                      resized_image.shape[0] - cropping_length]
    return cv2.resize(resized_image, (target_height, target_width))


def preprocess_frame(image, target_height=224, target_width=224):
    image = resize_frame(image, target_height, target_width)
    image = skimage.img_as_float(image).astype(np.float32)
    # 根据在ILSVRC数据集上的图像的均值（RGB格式）进行白化
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image


def extract_features():
    # 读取视频列表，让视频按照id升序排列
    videos = sorted(os.listdir(video_root), key=video_sort_lambda)
    nvideos = len(videos)

    # 创建保存视频特征的hdf5文件
    if os.path.exists(feature_h5_path):
        # 如果hdf5文件已经存在，说明之前处理过，或许是没有完全处理完
        # 使用r+ (read and write)模式读取，以免覆盖掉之前保存好的数据
        h5 = h5py.File(feature_h5_path, 'r+')
        dataset_feats = h5[feature_h5_feats]
        dataset_lens = h5[feature_h5_lens]
    else:
        h5 = h5py.File(feature_h5_path, 'w')
        dataset_feats = h5.create_dataset(feature_h5_feats,
                                          (nvideos, max_frames, feature_size),
                                          dtype='float32')
        dataset_lens = h5.create_dataset(feature_h5_lens, (nvideos,), dtype='int')

    for i, video in enumerate(videos):
        print(video, end=' ')
        video_path = os.path.join(video_root, video)
        # 提取视频帧以及视频小块
        frame_list, clip_list, frame_count = sample_frames(video_path, train=True)
        print(frame_count)

        # 把图像做一下处理，然后转换成（batch, channel, height, width）的格式
        frame_list = np.array([preprocess_frame(x) for x in frame_list])
#         frame_list = frame_list.transpose((0, 3, 1, 2))
#         frame_list = Variable(torch.from_numpy(frame_list), volatile=True).cuda()

        # 视频特征的shape是max_frames x (2048 + 4096)
        # 如果帧的数量小于max_frames，则剩余的部分用0补足
        feats = np.zeros((max_frames, feature_size), dtype='float32')

        # 先提取表观特征
        af = extract_res(frame_list)

        # 再提取动作特征
        clip_list = np.array([[resize_frame(x, 112, 112)
                               for x in clip] for clip in clip_list])
#         clip_list = clip_list.transpose(0, 4, 1, 2, 3).astype(np.float32)
#         clip_list = Variable(torch.from_numpy(clip_list), volatile=True).cuda()
        mf = extract_c3d(clip_list)
        
        # 合并表观和动作特征
        #feats[:frame_count, :] = torch.cat([af, mf], dim=1).data.cpu().numpy()
#         print(af)
#         print('ljyy')
#         print(mf)
        feats[:frame_count, :] =  np.concatenate((af, mf), axis=1)
        print(feats[:frame_count, :].shape)
        dataset_feats[i] = feats
        dataset_lens[i] = frame_count

def extract_c3d(frame_list):
    """Extract c3d features."""
    # Session config.
#     sess_config = tf.ConfigProto()
#     sess_config.gpu_options.allow_growth = True
#     sess_config.gpu_options.visible_device_list = '0'

    gpu_fraction = 0.1
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        extractor = VideoC3DExtractor(max_frames, sess)
        c3d_features = extractor.extract(frame_list)
        
    return c3d_features

def extract_res(clip_list):
    """Extract res features."""
    # Session config.
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
#     sess_config.gpu_options.visible_device_list = '0'

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
        extractor = VideoResExtractor(max_frames, sess)
        res_features = extractor.extract(clip_list)
        
    return res_features


def main():
#     aencoder = VideoC3DExtractor()
#     aencoder.eval()
#     aencoder.cuda()

#     mencoder = VideoResExtractor
#     mencoder.eval()
#     mencoder.cuda()

    extract_features()


if __name__ == '__main__':
    main()
