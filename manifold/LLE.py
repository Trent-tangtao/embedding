import numpy as np



class ManifoldData():
    def __init__(self):
        pass

    def spiral3D(self, begin_circle, end_circle, num_points ,a = 1.0, b = 1.0,):
        theta = np.random.uniform(low = 2 * begin_circle * np.pi,
                                  high = 2 * end_circle * np.pi,
                                  size = (num_points, ))
        radius = a + b * theta
        xs = radius * np.cos(theta)
        ys = radius * np.sin(theta)
        zs = np.random.uniform(low = 0.0, high = 10.0, size = (num_points,))
        return [ys, zs, xs]

ma = ManifoldData()
data1 = ma.spiral3D(begin_circle = 0.0, end_circle = 0.5, num_points = 5000)
data2 = ma.spiral3D(begin_circle = 0.5, end_circle = 1.0, num_points = 5000)
data3 = ma.spiral3D(begin_circle = 1.0, end_circle = 1.5, num_points = 5000)


class LLE():
    def __init__(self, k_neighbors, low_dims):
        '''
        init function
        @params k_neighbors : the number of neigbors
        @params low_dims : low dimension
        '''
        self.k_neighbors = k_neighbors
        self.low_dims = low_dims

    def fit_transform(self, X):
        '''
        transform X to low-dimension
        @params X : 2-d numpy.array (n_samples, high_dims)
        '''
        n_samples = X.shape[0]
        # calculate pair-wise distance
        dist_mat = pairwise_distances(X)
        # index of neighbors, not include self
        neighbors = np.argsort(dist_mat, axis=1)[:, 1: self.k_neighbors + 1]

        # neighbor combination matrix
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            mat_z = X[i] - X[neighbors[i]]  # x - \eta_i
            mat_c = np.dot(mat_z, mat_z.transpose())  # C
            w = np.linalg.solve(mat_c, np.ones(mat_c.shape[0]))
            W[i, neighbors[i]] = w / w.sum()

        # sparse matrix M
        I_W = np.eye(n_samples) - W
        M = np.dot(I_W.transpose(), I_W)

        # solve the d+1 lowest eigen values
        eigen_values, eigen_vectors = np.linalg.eig(M)
        index = np.argsort(eigen_values)[1: self.low_dims + 1]
        selected_eig_values = eigen_values[index]
        selected_eig_vectors = eigen_vectors[index]

        self.eig_values = selected_eig_values
        self.low_X = selected_eig_vectors.transpose()
        print(self.low_X.shape)
        return self.low_X
