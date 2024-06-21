import os
import gzip
import struct
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader

def load_image_labels(data_dir,image_file,label_file):
    image_path = os.path.join(data_dir, image_file)
    label_path = os.path.join(data_dir, label_file)
    with gzip.open(label_path) as f_label:
        magic, num = struct.unpack('>II', f_label.read(8))
        label = np.fromstring(f_label.read(), dtype=np.int8)


    with gzip.open(image_path, 'rb') as f_image:
        magic, num, rows, cols = struct.unpack('>IIII', f_image.read(16))
        image = np.fromstring(f_image.read(), dtype=np.uint8).\
            reshape((len(label), rows, cols))
    return image,label


def load_data(data_dir,image_file,label_file,batch_size=128):

    train_images, train_labels = \
        load_image_labels(data_dir, image_file, label_file)

    # print(train_images.shape, train_labels.shape)

    train_dataset = TensorDataset(torch.tensor(train_images).float(),
                                  torch.tensor(train_labels).long())
    train_loader = DataLoader(dataset=train_dataset, shuffle=True,
                              batch_size=batch_size)
    return train_loader


    return train_loader

def show_image(image,label):
    plt.imshow(image,cmap='gray')
    plt.title(label)
    plt.show()