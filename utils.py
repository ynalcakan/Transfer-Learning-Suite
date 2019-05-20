# Layers
from keras.layers import Dense, Activation, Flatten, Dropout
from keras import backend as K

# Other
from keras import optimizers
from keras import losses
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model

# Utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random, glob
import os, sys, csv
import cv2
import time, datetime

import split_folders

import io
import itertools
from packaging import version
from six.moves import range
import tensorflow as tf
import sklearn.metrics

# Keras-vis packages
import matplotlib.cm as cm
from vis.visualization import visualize_cam, visualize_activation, get_num_filters
from vis.input_modifiers import Jitter
from vis.utils import utils


def _split_folders(input_folder):
    try:
        split_folders.ratio(input_folder, output="output", seed=7, ratio=(0.6, 0.2, 0.2)) # default values
        print("done!")
    except Exception as e:
        print(e)


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def visualize_attention(model, layer_name, penultimate_layer, image_list):
    layer_idx = utils.find_layer_idx(model, layer_name)
    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    for modifier in [None, 'guided', 'relu']:
        plt.figure()
        f, ax = plt.subplots(1, 2)
        plt.suptitle("vanilla" if modifier is None else modifier)
        for i, img in enumerate(image_list):
            # 20 is the imagenet index corresponding to `ouzel`
            grads = visualize_cam(model, layer_idx, filter_indices=20,
                                  seed_input=img, penultimate_layer_idx=penultimate_layer,
                                  backprop_modifier=modifier)
            # Lets overlay the heatmap onto original image.
            jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
            ax[i].imshow(overlay(jet_heatmap, img))


def _visualize_activation(model, layer_name, max_iter, class_number):
    layer_idx = utils.find_layer_idx(model, layer_name)
    img = visualize_activation(model, layer_idx, max_iter=max_iter, filter_indices=class_number, input_modifiers=[Jitter(16)])
    plt.imshow(img)


def visualize_conv_filter(model, layer_name):
    layer_idx = utils.find_layer_idx(model, layer_name)
    # Visualize all filters in this layer.
    filters = np.arange(get_num_filters(model.layers[layer_idx]))

    # Generate input image for each filter.
    vis_images = []
    for idx in filters:
        img = visualize_activation(model, layer_idx, filter_indices=idx, input_modifiers=[Jitter(0.05)])

        # Utility to overlay text on image.
        img = utils.draw_text(img, 'Filter {}'.format(idx))
        vis_images.append(img)

    # Generate stitched image palette with 8 cols.
    stitched = utils.stitch_images(vis_images, cols=8)
    plt.axis('off')
    plt.imshow(stitched)
    plt.title(layer_name)
    plt.show()


def save_class_list(class_list, model_name, dataset_name):
    class_list.sort()
    target=open("./checkpoints/" + model_name + "_" + dataset_name + "_class_list.txt",'w+')
    for c in class_list:
        target.write(c)
        target.write("\n")

def load_class_list(class_list_file):
    class_list = []
    with open(class_list_file, 'r') as csvfile:
        file_reader = csv.reader(csvfile)
        for row in file_reader:
            class_list.append(row)
    class_list.sort()
    return class_list

# Get a list of subfolders in the directory
def get_subfolders(directory):
    subfolders = os.listdir(directory)
    subfolders.sort()
    return subfolders

# Get number of files by searching directory recursively
def get_num_files(directory):
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt

# Add on new FC layers with dropout for fine tuning
def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False
    """
    ResNet50 finetune last Conv layer
    for layer in base_model.layers:
        if layer.name == "conv2d_94" or "batch_normalization_94":
            layer.trainable = True
        else:
            layer.trainable = False
    """

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x) # New FC layer, random init
        x = Dropout(dropout)(x)

    predictions = Dense(num_classes, activation='softmax')(x) # New softmax layer

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

# Plot the training and validation loss + accuracy
def plot_training(history, model_name, class_string, today):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    fig1 = plt.gcf()
    fig1.savefig("/home/yagiz/Sourcebox/git/Transfer-Learning-Suite/{}loss_acc_graphs/{}_accuracy_{}".format(class_string,model_name, today))
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    fig2 = plt.gcf()
    fig2.savefig("/home/yagiz/Sourcebox/git/Transfer-Learning-Suite/{}loss_acc_graphs/{}_loss_{}".format(class_string,model_name, today))
    #plt.show()
