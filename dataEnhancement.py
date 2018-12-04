import os
import moviepy
import random
from moviepy.editor import *

path="/disk/private-data/why/ZhiJiang_VQA/data/train"
files= os.listdir(path)
random.shuffle(files)
for i,file in enumerate(files):
    clip=VideoFileClip(path+"/"+file)
    if i%4==0:
        clip=moviepy.video.fx.all.mirror_x(clip)
    elif i%4==1:
        clip=moviepy.video.fx.all.colorx(clip,1)
    elif i%4==2:
        clip=moviepy.video.fx.all.colorx(clip,1.8)
    clip.write_videofile("/disk/private-data/why/ZhiJiang_VQA/data/train_new"+"/"+file)
    print(file)

print("finished")