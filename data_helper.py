import numpy as np
import os

num_classes = 10
num_digits = 2

def load_dataset(dataset_path='synthetic_dataset', zca_whitening=True):
    """Load numpy format dataset stored in a specific path."""
    images_1 = np.load(os.path.join(dataset_path, "prev_images.npy")).astype('float32')
    images_2 = np.load(os.path.join(dataset_path, "images.npy")).astype('float32')
    delta_ps = np.load(os.path.join(dataset_path, "delta_ps.npy")).astype('float32')
    return images_1, images_2, delta_ps

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]