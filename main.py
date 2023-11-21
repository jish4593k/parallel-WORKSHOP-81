import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from collections import namedtuple


class KMeansSequencer:
    def __init__(self):
        self.cluster_size = 0
        self.file_path = ""
        self.threshold = 0.0
        self.training_data = np.array([])
        self.num_of_data = 0
        self.cluster_index = np.array([])
        self.cluster = np.array([])
        self.membership = np.array([])
        self.random_sample = np.array([])

    def initialize(self, cluster_size, file_path, threshold):
        self.cluster_size = cluster_size
        self.file_path = file_path
        self.threshold = threshold
        self.training_data = self.load_data()
        self.num_of_data = self.training_data.shape[0]
        self.cluster_index = np.random.choice(self.num_of_data, self.cluster_size, replace=False)
        self.cluster = self.training_data[self.cluster_index]
        self.membership = np.full(self.num_of_data, -1)
        self.random_sample = np.random.choice(self.num_of_data, 10000, replace=False)

    def load_data(self):
        mat = scipy.io.loadmat(self.file_path)
        data = mat['images'].astype(int)
        num_of_data = data.shape[2]
        reshaped_data = data.transpose(2, 0, 1).reshape(num_of_data, -1)
        return reshaped_data  # (60000, 784)

    def run_kmeans(self):
        loop = 0
        while True:
            print('loop' + str(loop))
            loop += 1
            cluster_map = {j: [] for j in range(self.cluster.shape[0])}
            delta = self.update_clusters(cluster_map)
            self.cluster = self.calculate_centers(cluster_map)
            print(delta)
            if delta / self.training_data.shape[0] < self.threshold or loop > 500:
                break

    def update_clusters(self, cluster_map):
        delta = 0
        for i in range(self.training_data.shape[0]):
            index = self.find_cluster(self.training_data[i])
            if self.membership[i] != index:
                self.membership[i] = index
                delta += 1
            try:
                cluster_map[index].append(self.training_data[i])
            except KeyError:
                cluster_map[index] = [self.training_data[i]]
        return delta

    def find_cluster(self, current_data):
        distances = tf.norm(current_data - self.cluster, axis=1)
        index = tf.argmin(distances)
        return index.numpy()

    def calculate_centers(self, cluster_map):
        new_cluster = np.zeros_like(self.cluster)
        for i in range(self.cluster.shape[0]):
            new_cluster[i, :] = np.mean(cluster_map[i], axis=0)
        return new_cluster

    def plot_clusters(self):
        fig, axs = plt.subplots(nrows=1, ncols=self.cluster.shape[0])
        plt_cluster = self.cluster.reshape(self.cluster.shape[0], 28, 28)
        for i in range(self.cluster.shape[0]):
            axs[i].imshow(plt_cluster[i])
            axs[i].axis("off")
        plt.show()


if __name__ == '__main__':
    sequencer = KMeansSequencer()
    sequencer.initialize(10, './mnist_data/images.mat', threshold=0.001)
    sequencer.run_kmeans()
    sequencer.plot_clusters()
