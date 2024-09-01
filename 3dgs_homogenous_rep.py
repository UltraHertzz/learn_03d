import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class GaussianSphere:

    def __init__(self, mean, rot, scale):
        self.mean = mean
        self.covariance = self.build_covariance(rot, scale)
        self.homogeneous_gaussian = self.build_homogeneous_gaussian(mean, self.covariance)

    def build_covariance(self, rot, scale):
        L = np.dot(rot, np.diag(scale))
        R = np.dot(np.diag(scale), rot.T)
        return np.dot(L, R)
    

    def build_homogeneous_gaussian(self, mean, covariance):

        self.homogeneous_gaussian = np.zeros((4, 4))
        self.homogeneous_gaussian[:3, :3] = covariance
        self.homogeneous_gaussian[:3, 3] = mean
        self.homogeneous_gaussian[3, 3] = 1

        return self.homogeneous_gaussian
    
    def sample(self, n_samples):
        return np.random.multivariate_normal(self.mean, self.covariance, n_samples)
    
    def trans_homo(self, transform=np.eye(4)):
        rot = transform[:3, :3]
        trans_gaussian = transform @ self.homogeneous_gaussian @ rot.T
        return trans_gaussian
    
    def affine_trans(self, affine_matrix=np.eye(4)):
        A = affine_matrix[:3,:3]
        b = affine_matrix[:3, 3]
        trans_mean = A @ self.mean + b
        trans_cov = A @ self.covariance @ A.T
        return trans_mean, trans_cov
    
# Define the scaling matrix S
S = np.array([2, 1, 1.5])  # Scaling factors along x, y, z axes


# Define the rotation matrix R (example: a rotation around z-axis by 45 degrees)
theta = np.pi / 4  # 45 degrees
R = np.array([[np.cos(theta), -np.sin(theta), 0],
              [np.sin(theta), np.cos(theta), 0],
              [0, 0, 1]])

gaussian = GaussianSphere(mean=[0, 0, 0], rot=R, scale=S)


# Function to plot the 3D Gaussian ellipsoid
def plot_gaussian_ellipsoid(mean, cov, ax=None, **kwargs):
    """Plot a 3D Gaussian as an ellipsoid."""
    if ax is None:
        ax = plt.gca(projection='3d')

    # Find and sort eigenvalues and associated eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Radii of the ellipsoid
    rx, ry, rz = np.sqrt(eigvals)

    # Calculate the rotation matrix
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    # Rotate data with orientation matrix
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], eigvecs.T)

    x += mean[0]
    y += mean[1]
    z += mean[2]

    ax.plot_surface(x, y, z, **kwargs)

# Plot the Gaussian distribution as an ellipsoid
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

plot_gaussian_ellipsoid(gaussian.mean, gaussian.covariance, ax=ax, rstride=4, cstride=4, color='c', alpha=0.5)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Gaussian Distribution')

plt.show()