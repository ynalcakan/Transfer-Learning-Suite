import os
import time
from termcolor import colored

train_command = "python main.py --num_epochs 1 --mode 'train' --dataset {} --h_flip 1 --v_flip 1 --rotation 0.25 --zoom 0.3 --model {} "

model_list = ["VGG16", "VGG19", "MobileNet", "MobileNetV2", "InceptionV3",
              "InceptionResNetV2", "Xception", "DenseNet121", "DenseNet169",
              "DenseNet201", "NASNetMobile", "NASNetLarge","ResNet50"]

# get results for 2 classification
for model in model_list:
    try:
        os.system(train_command.format("dataset_w_2class", model))
    except Exception as e:
        print(colored(e, 'red'))
        pass
