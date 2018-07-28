import gzip
import os

from scipy.misc import imsave
import numpy as np

IMAGE_SIZE = 28


def extract_data(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return data


def extract_labels(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


train_data_filename = 'train-images.gz'
train_labels_filename = 'train-labels.gz'
test_data_filename = 'test-images.gz'
test_labels_filename = 'test-labels.gz'

# -----
# process the train data
train_data = extract_data(train_data_filename, 60000)
train_labels = extract_labels(train_labels_filename, 60000)
# create test directories
for i in range(10):
    os.makedirs("train/{:d}".format(i), exist_ok=True)
# save the .jpg on the correct directory
for i in range(len(train_data)):
    imsave('train/{:d}/{:d}.jpg'.format(train_labels[i], i), train_data[i][:, :, 0])

# -----
# process the test data
test_data = extract_data(test_data_filename, 10000)
test_labels = extract_labels(test_labels_filename, 10000)
# create test directories
for i in range(10):
    os.makedirs("valid/{:d}".format(i), exist_ok=True)
# save the .jpg on the correct directory
for i in range(len(test_data)):
    imsave('valid/{:d}/{:d}.jpg'.format(test_labels[i], i), test_data[i][:, :, 0])

