import tensorflow as tf
import sys
sys.path.append("..")
from util.c3d import c3d
import inspect
import os
import numpy as np

class VideoC3DExtractor(object):
    """Select uniformly distributed clips and extract its C3D feature."""

    def __init__(self, clip_num, sess):
        """Load C3D model."""
        self.clip_num = clip_num
        self.inputs = tf.placeholder(
            tf.float32, [self.clip_num, 16, 112, 112, 3])
        self.c3d_features = c3d(self.inputs, 1, clip_num)
        saver = tf.train.Saver()
        path = inspect.getfile(inspect.currentframe())  #inspect.getfile定义了名称
        path = os.path.abspath(os.path.join(path, os.pardir))  #os.pardir引用父目录
        #/home/dian/yy/BD/util
        saver.restore(sess, os.path.join(
            path, 'sports1m_finetuning_ucf101.model'))
        self.mean = np.load(os.path.join(path, 'crop_mean.npy'))
        self.sess = sess

    def extract(self, clips):
        """Get 4096-dim activation as feature for video.
        Args:
            path: Path of video.
        Returns:
            feature: [self.batch_size, 4096]
        """
        #clips = self._select_clips(path)
        feature = self.sess.run(
            self.c3d_features, feed_dict={self.inputs: clips})
        return feature
    
class VideoResExtractor(object):
    """Select uniformly distributed clips and extract its Res feature."""

    def __init__(self, clip_num, sess):
        """Load RES model."""
        self.clip_num = clip_num
        tf.keras.backend.set_session(sess)
        self.inputs = tf.placeholder(tf.float32,[self.clip_num, 224, 224, 3] )
        self.model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet',
                                                            input_shape=(
                                                                224, 224, 3),
                                                            pooling='avg')
        #self.res_features = self.model(self.inputs)
        #saver = tf.train.Saver()
        #path = inspect.getfile(inspect.currentframe())  #inspect.getfile定义了名称
        #path = os.path.abspath(os.path.join(path, os.pardir))  #os.pardir引用父目录
        #/home/dian/yy/BD/util
        #saver.restore(sess, os.path.join(
        #    path, 'resnet50_.model'))
        #self.mean = np.load(os.path.join(path, 'res_crop_mean.npy'))
        #self.sess = sess

    def extract(self, clips):
        """Get 4096-dim activation as feature for video.
        Args:
            path: Path of video.
        Returns:
            feature: [self.batch_size, 4096]
        """
        #clips = self._select_clips(path)
        feature = self.model.predict(clips)
        #feature = self.sess.run(
        #    self.res_features, feed_dict={self.inputs: clips})
        return feature