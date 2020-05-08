# coding: utf-8
import skvideo.io
import numpy as np
from PIL import Image
import os
import json


def extract_video_frame(video_file_path="/disk/ZhiJiang_VideoQA_Pytorch/data/processed/video/", extract_image_path="/disk/video_qa_B_40/"):
    """
    extract video frames to images
    """
    split_dict = {}
    if not os.path.exists(extract_image_path):
        os.mkdir(extract_image_path)
    files = os.listdir(video_file_path)
    a = len(files)
    b = 1
    for f in files:
        try:
            split_dict[f.split('.mp4')[0]] = []
            video_data = skvideo.io.vread(os.path.join(video_file_path ,f))
            length = len(video_data)
            j = 1
            for i in list(list(np.linspace(0, length-1, num=40, endpoint=True, retstep=False, dtype=np.int64))):
                img = Image.fromarray(video_data[i])
                img.save(os.path.join(extract_image_path, f.split(".mp4")[0] + "_" + str(j) + ".png"), "PNG")
                split_dict[f.split('.mp4')[0]].append(f.split(".mp4")[0] + "_" + str(j) + ".png")
                j += 1
            print(str(b) + "/" + str(a))
            b += 1
            
        except ValueError:
            print(f)

    json.dump(split_dict, open('../data/processed/video_img_split.json', 'w'))

if __name__ == '__main__':
    extract_video_frame()
