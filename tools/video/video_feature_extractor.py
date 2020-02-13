import argparse
import cv2
import logging
import os
import tempfile
import threading

from multiprocessing.pool import ThreadPool as Pool

from PIL import Image

import numpy as np

import tensorflow as tf

from tensorflow.keras.applications import vgg19, mobilenet_v2, inception_v3
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('tensorflow')
logger.propagate = False

global_graph = tf.get_default_graph()
global_session = tf.Session(graph=global_graph)


class ImageFeatureExtraction():
    """
    A class that builds a TF graph with a pre-trained image model (on imagenet)
    Also takes care of preprocessing. Input should be a regular RGB image (0-255)

    Reference:
    https://github.com/zachmoshe/zachmoshe.com/blob/master/content/use-keras-models-with-tf/vgg19.py
    """

    def __init__(self,
                 extractor,
                 image_shape,
                 input_tensor=None):
        self.image_shape = image_shape

        self.extractor = extractor
        self.feature_extractors = {
            'MobileNetV2': {
                'loader': self.__mobilenetv2_loader,
                'preprocessing': self.__mobilenetv2_preprocessing,
            },
            'InceptionV3': {
                'loader': self.__inceptionv3_loader,
                'preprocessing': self.__inceptionv3_preprocessing,
            },
            'VGG19': {
                'loader': self.__vgg19_loader,
                'preprocessing': self.__vgg19_preprocessing,
            },
            # implement other image feature extraction below
        }
        if self.extractor not in self.feature_extractors:
            tf.logging.error('%s model is not supportted in ImageFeatureExtraction. Choose \
                             among %s' % (self.extractor, ', '.join(self.feature_extractors.keys())))
            exit

        self.preprocessor = self.feature_extractors[self.extractor]['preprocessing']
        self.loader = self.feature_extractors[self.extractor]['loader']

        self._build_graph(input_tensor)

    def __vgg19_loader(self, img):
        model = tf.keras.applications.vgg19.VGG19(
            weights='imagenet',
            include_top=False,
            input_tensor=img,
        )
        model.trainable = False  # freeze all model weights

        return model

    def __vgg19_preprocessing(self, input_tensor):
        RGB_MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape(
            (1, 1, 1, 3)).astype(np.float32)
        img = input_tensor - RGB_MEAN_PIXELS
        img = tf.reverse(img, axis=[-1])

        return img

    def __inceptionv3_loader(self, img):
        model = tf.keras.applications.inception_v3.InceptionV3(
            weights='imagenet',
            include_top=False,
            input_tensor=img,
        )
        model.trainable = False

        return model

    def __inceptionv3_preprocessing(self, input_tensor):
        return input_tensor

    def __mobilenetv2_loader(self, img):
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_tensor=img,
        )
        model.trainable = False

        return model

    def __mobilenetv2_preprocessing(self, input_tensor):
        # normalize image into [-1:1] range
        return (2.0 / 255.0) * input_tensor - 1.0

    def _build_graph(self, input_tensor):
        with tf.Session() as sess:
            with tf.variable_scope(self.extractor):
                with tf.name_scope('inputs'):
                    if input_tensor is None:
                        input_tensor = tf.placeholder(tf.float32,
                                                      shape=self.image_shape,
                                                      name='input_img')
                    else:
                        assert self.image_shape == input_tensor.shape
                    self.input_tensor = input_tensor

                with tf.name_scope('preprocessing'):
                    img = self.preprocessor(self.input_tensor)

                with tf.variable_scope('model'):
                    self.model = self.loader(img)

                self.outputs = {l.name: l.output for l in self.model.layers}

            self.model_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                   scope='%s/model' % self.extractor)

            with tempfile.NamedTemporaryFile() as f:
                self.tf_checkpoint_path = tf.train.Saver(
                    self.model_weights).save(sess, f.name)

        self.model_weights_tensors = set(self.model_weights)

    def load_weights(self):
        sess = tf.get_default_session()
        tf.train.Saver(self.model_weights).restore(
            sess, self.tf_checkpoint_path)

    def __getitem__(self, key):
        return self.outputs[key]


def create_features(imagePath, pre_model):
    """
     Args:
        imagePath: string, path to an image file e.g., '/path/to/example.JPG'.
        pre_model: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
        feature: 2D tensor of image feature
    """
    def __load_image(img_path, output_size):
        img = Image.open(img_path)
        img = img.resize(output_size[::-1])
        return np.asarray(img, dtype=np.float32)

    def __reshape_feature_map(feature_map):
        shape = list(feature_map.shape)
        assert len(shape) == 4, 'Feature map requires a 4D tensor'

        # If feature map is 4D tensor (batch_size, 8, 8, 2048)
        # we map it to (batch_size, 64, 2048)
        feature_map = tf.reshape(
            feature_map, [-1, shape[1] * shape[2], shape[3]])

        # return a 3D tensor since the batch size is always 1
        return tf.squeeze(feature_map)

    tf.logging.info('Processing %s' % imagePath)

    image = __load_image(imagePath, output_size=pre_model['img_shape'])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pre_model['model'].load_weights()
        fd = {pre_model['my_img']: np.expand_dims(image, 0)}

        feature = sess.run(pre_model['model_output'], fd)
        feature = __reshape_feature_map(feature).eval()

    return feature


def __get_image_coder(pretrained_model='VGG19'):
    tf.reset_default_graph()

    model_info = {
        'VGG19': {
            'last_layer': 'block5_pool',
            'default_shape': (224, 224)
        },
        'InceptionV3': {
            'last_layer': 'mixed10',
            'default_shape': (299, 299)
        },
        'MobileNetV2': {
            'last_layer': 'out_relu',
            'default_shape': (192, 192)
        }
        # implement other image feature extraction below
    }
    if pretrained_model not in model_info:
        tf.logging.error('%s model is not supportted in ImageFeatureExtraction. Choose \
                         among %s' % (pretrained_model, ', '.join(model_info.keys())))
        exit

    img_shape = model_info[pretrained_model]['default_shape']
    last_layer = model_info[pretrained_model]['last_layer']

    my_img = tf.placeholder(tf.float32,
                            (1, img_shape[0], img_shape[1], 3),
                            name='my_original_image')

    model = ImageFeatureExtraction(
        extractor=pretrained_model,
        image_shape=(1, img_shape[0], img_shape[1], 3),
        input_tensor=my_img
    )
    model_output = tf.identity(model[last_layer], name='my_output')

    return {
        'model': model,
        'img_shape': img_shape,
        'my_img': my_img,
        'model_output': model_output
    }


def extract_image_feature(thread_index, ranges, image_list, image_model):
    images = image_list[ranges[thread_index][0]:ranges[thread_index][1]]
    # each thread will have its own image coder
    coder = __get_image_coder(image_model)

    for image in images:
        image_feature_name = '%s.npy' % str(image)

        tf.logging.info('[thread %d]: Image: %s' %
                        (thread_index, image_feature_name))

        # extract image features
        features = create_features(image, coder)
        np.save(image_feature_name, features)


def extract_image(video_file, output_dir='data'):
    tf.logging.info('Processing %s' % video_file)
    cam = cv2.VideoCapture(video_file)  # Read the video from specified path

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # creating a output folder
    except OSError:
        tf.logging.error('Fail on creating directory %s' % output_dir)

    currentframe = 0  # frame count

    image_list = []
    while(True):
        ret, frame = cam.read()  # reading each frame
        if ret:
            # if video is still left continue creating images
            image_name = 'frame.%s.jpg' % str(currentframe)
            image_path = os.path.join(output_dir, image_name)

            tf.logging.info('Creating ... ' + image_path)
            cv2.imwrite(image_path, frame)  # writing the extracted images

            image_list.append(image_path)
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

    return image_list


def process_video(video_file,
                  output_dir='data',
                  num_threads=4,
                  image_model='VGG19'):
    image_list = extract_image(video_file, output_dir)

    spacing = np.linspace(0, len(image_list), num_threads + 1).astype(np.int)
    ranges = []
    threads = []

    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()
    for thread_index in range(len(ranges)):
        args = (thread_index,
                ranges,
                image_list,
                image_model,)
        t = threading.Thread(
            target=extract_image_feature, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)


def main():
    parser = argparse.ArgumentParser(
        description="Extract images and their features from a video clip using pretrained image models.")
    parser.add_argument("-i", type=str, required=True,
                        help="path to an input video file")
    parser.add_argument("-o", type=str, default='data',
                        help="path to an output directory that will store images and their features")
    parser.add_argument("-t", type=int, default=4,
                        help="Number of thread for image feature processing")
    parser.add_argument("-m", type=str, default='VGG19',
                        help="pretrained image model, currently support VGG19, InceptionV3, MobileNetV2")

    args = parser.parse_args()
    process_video(args.i, args.o, args.t, args.m)


if __name__ == "__main__":
    main()
