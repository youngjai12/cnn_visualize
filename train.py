import tensorflow as tf
from model import URLNet
import os
import time

def grad_cam(model: URLNet):
    grads = tf.gradients(model.score_layer[:, 0], model.word_feature_maps)
    # K.gradients(model.output[:,0], layers_wt)[0]

    # 필터별로 가중치를 구한다 ?
    # pooled_grads = tf.mean(grads, axis=(0, 1))
    # get_pooled_grads = K.function([model.input, model.sample_weights[0], K.learning_phase()],
    #                               [pooled_grads, layers.output[0]])

    model = URLNet()
    model.build().output

