import torch
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Gaussian:

    def __init__(self, center, rot=np.array([1,0,0,0]), scale=np.array([1,1,1]), cov=None, build_from_cov=False) -> None:
        self.center = center
        if not build_from_cov:
            self.rot = rot # q
            self.scale = scale
            self.R = self.build_rotational_matrix_from_quaternion()
            self.S = self.build_scale_matrix()
            self.S_inv = self.get_invserse_scale_matrix()

            self.cov = self.build_cov()
        else:
            if cov is None:
                raise ValueError('cov should be provided')
            self.cov = cov


    def get_exp_scale_matrix(self):
        S_exp = np.diag([np.exp(self.scale[0]), np.exp(self.scale[1]), np.exp(self.scale[2])])
        return S_exp

    def get_invserse_scale_matrix(self):
        S_inv = np.diag([1/self.scale[0], 1/self.scale[1], 1/self.scale[2]])
        return S_inv

    def build_rotational_matrix_from_quaternion(self):
        q = self.rot
        w, x, y, z = q[0], q[1], q[2], q[3]
        R = np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                           [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                           [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])
        return R
    
    def build_scale_matrix(self):
        S = np.diag([self.scale[0], self.scale[1], self.scale[2]])
        return S

    def build_cov(self):

        L = np.dot(self.R, self.S)
        cov = np.dot(L, L.T)
        return cov

    def build_gaussian_from_mean_cov(self):
        data = np.random.multivariate_normal(self.center, self.cov, 100)
        return data
    
    def plot_gaussian(self):
        data = self.build_gaussian_from_mean_cov()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(data[:, 0], data[:, 1], data[:, 2])

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        xyz = np.array([x.flatten(), y.flatten(), z.flatten()])

        xyz_transformed = self.S @ np.diag([np.e,np.e,np.e]) @ xyz
        xyz_transformed = self.R @ xyz_transformed

        x_transformed = xyz_transformed[0, :].reshape(x.shape) + self.center[0]
        y_transformed = xyz_transformed[1, :].reshape(y.shape) + self.center[1]
        z_transformed = xyz_transformed[2, :].reshape(z.shape) + self.center[2]

        ax.plot_surface(x_transformed, y_transformed, z_transformed, color='b', alpha=0.2)

        plt.show()
    
    @staticmethod
    def plot_multi_gaussian(gaussians, conv_gaussian=None): 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for gaussian in gaussians:
            data = gaussian.build_gaussian_from_mean_cov()
            color = np.random.choice(colors)
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color)

            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones_like(u), np.cos(v))

            xyz = np.array([x.flatten(), y.flatten(), z.flatten()])

            xyz_transformed = gaussian.S @ np.diag([np.e,np.e,np.e]) @ xyz
            xyz_transformed = gaussian.R @ xyz_transformed

            x_transformed = xyz_transformed[0, :].reshape(x.shape) + gaussian.center[0]
            y_transformed = xyz_transformed[1, :].reshape(y.shape) + gaussian.center[1]
            z_transformed = xyz_transformed[2, :].reshape(z.shape) + gaussian.center[2]

            ax.plot_surface(x_transformed, y_transformed, z_transformed, color=color, alpha=0.2)
        if conv_gaussian is not None:
            data = conv_gaussian.build_gaussian_from_mean_cov()
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r')

        plt.show()

    @staticmethod
    def gaussian_conv(gaussians):
        mean = np.mean([gaussian.center for gaussian in gaussians], axis=0)
        cov = np.sum([gaussian.cov for gaussian in gaussians], axis=0)
        return Gaussian(mean, cov=cov, build_from_cov=True)

            

if __name__ == '__main__':
    center = np.array([0, 10, 10])
    rot = np.array([0, 1, 0, 0])
    scale = np.array([1, 1, 1])
    # gaussian = Gaussian(center, rot, scale)
    # gaussian.plot_gaussian()
    gaussians = []

    for _ in range(3):
        center = np.random.uniform(-10, 10, size=3)
        rot = np.random.uniform(-1, 1, size=4)
        scale = np.random.uniform(-1, 1, size=3)
        gaussian = Gaussian(center, rot, scale)
        gaussians.append(gaussian)
    conv_gaussian = Gaussian.gaussian_conv(gaussians)
    Gaussian.plot_multi_gaussian(gaussians)