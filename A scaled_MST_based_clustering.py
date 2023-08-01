import numpy as np
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from skimage import io

def mst_clustering(image, n_clusters):
    # Create a distance matrix from the image
    distance_matrix = np.zeros((image.shape[0] * image.shape[1], image.shape[0] * image.shape[1]))
    for i in range(image.shape[0] * image.shape[1]):
        for j in range(i + 1, image.shape[0] * image.shape[1]):
            x1, y1 = i // image.shape[1], i % image.shape[1]
            x2, y2 = j // image.shape[1], j % image.shape[1]
            distance = np.sqrt(np.sum((image[x1, y1, :] - image[x2, y2, :]) ** 2))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    # Build a minimum spanning tree using Prim's algorithm
    mst = np.zeros(distance_matrix.shape)
    visited = np.zeros(distance_matrix.shape[0], dtype=bool)
    visited[0] = True
    for i in range(1, distance_matrix.shape[0]):
        indices = np.where(visited)[0]
        nearest = np.argmin(distance_matrix[indices, i])
        j = indices[nearest]
        mst[i, j] = distance_matrix[i, j]
        mst[j, i] = distance_matrix[j, i]
        visited[i] = True

    # Scale the edges of the MST based on the average edge length
    avg_edge_length = np.sum(mst) / (np.count_nonzero(mst) / 2)
    mst[mst > avg_edge_length] = avg_edge_length

    # Cluster the vertices using agglomerative clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=mst)
    labels = clustering.fit_predict(np.arange(distance_matrix.shape[0]).reshape((-1, 1)))

    # Reshape the labels into an image
    labels_image = labels.reshape((image.shape[0], image.shape[1]))

    return labels_image



image = io.imread('image.jpg')
n_clusters = 4

labels_image = mst_clustering(image, n_clusters)

plt.imshow(labels_image)
plt.show()