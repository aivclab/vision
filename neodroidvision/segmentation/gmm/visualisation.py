import os

import numpy
from matplotlib import cm
from matplotlib import patches, pyplot

__all__ = ["visualise_3d_gmm", "visualise_2D_gmm"]


def plot_sphere(
    w=0, c=(0, 0, 0), r=(1, 1, 1), sub_divisions=10, ax=None, sigma_multiplier=3
):
    """
    plot a sphere surface
    Input:
        c: 3 elements list, sphere center
        r: 3 element list, sphere original scale in each axis ( allowing to draw elipsoids)
        sub_divisions: scalar, number of subdivisions (subdivision^2 points sampled on the surface)
        ax: optional pyplot axis object to plot the sphere in.
        sigma_multiplier: sphere additional scale (choosing an std value when plotting gaussians)
    Output:
        ax: pyplot axis object
    """

    if ax is None:
        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection="3d")
    pi = numpy.pi
    cos = numpy.cos
    sin = numpy.sin
    phi, theta = numpy.mgrid[
        0.0 : pi : complex(0, sub_divisions), 0.0 : 2.0 * pi : complex(0, sub_divisions)
    ]
    x = sigma_multiplier * r[0] * sin(phi) * cos(theta) + c[0]
    y = sigma_multiplier * r[1] * sin(phi) * sin(theta) + c[1]
    z = sigma_multiplier * r[2] * cos(phi) + c[2]
    cmap = cm.ScalarMappable()
    cmap.set_cmap("jet")
    c = cmap.to_rgba(w)

    ax.plot_surface(x, y, z, color=c, alpha=0.2, linewidth=1)

    return ax


def visualise_3d_gmm(points, w, mu, std_dev, export=False):
    """
    plots points and their corresponding gmm model in 3D
    Input:
        points: N X 3, sampled points
        w: n_components, gmm weights
        mu: 3 X n_components, gmm means
        std_dev: 3 X n_components, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    """

    n_components = mu.shape[1]
    N = int(numpy.round(points.shape[0] / n_components))
    # Visualize data
    fig = pyplot.figure(figsize=(8, 8))
    axes = fig.add_subplot(111, projection="3d")
    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    axes.set_zlim([-1, 1])
    pyplot.set_cmap("Set1")
    colors = cm.Set1(numpy.linspace(0, 1, n_components))
    for i in range(n_components):
        idx = range(i * N, (i + 1) * N)
        axes.scatter(
            points[idx, 0], points[idx, 1], points[idx, 2], alpha=0.3, c=colors[i]
        )
        plot_sphere(w=w[i], c=mu[:, i], r=std_dev[:, i], ax=axes)

    pyplot.title("3D GMM")
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")
    axes.view_init(35.246, 45)
    if export:
        if not os.path.exists("images/"):
            os.mkdir("images/")
        pyplot.savefig("images/3D_GMM_demonstration.png", dpi=100, format="png")
    pyplot.show()


def visualise_2D_gmm(points, w, mu, std_dev, export=False):
    """
    plots points and their corresponding gmm model in 2D
    Input:
        points: N X 2, sampled points
        w: n_components, gmm weights
        mu: 2 X n_components, gmm means
        std_dev: 2 X n_components, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    """
    n_components = mu.shape[1]
    N = int(numpy.round(points.shape[0] / n_components))
    # Visualize data
    fig = pyplot.figure(figsize=(8, 8))
    axes = pyplot.gca()
    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    pyplot.set_cmap("Set1")
    colors = cm.Set1(numpy.linspace(0, 1, n_components))
    for i in range(n_components):
        idx = range(i * N, (i + 1) * N)
        pyplot.scatter(points[idx, 0], points[idx, 1], alpha=0.3, c=colors[i])
        for j in range(8):
            axes.add_patch(
                patches.Ellipse(
                    mu[:, i],
                    width=(j + 1) * std_dev[0, i],
                    height=(j + 1) * std_dev[1, i],
                    fill=False,
                    color=[0.0, 0.0, 1.0, 1.0 / (0.5 * j + 1)],
                )
            )
        pyplot.title("GMM")
    pyplot.xlabel("X")
    pyplot.ylabel("Y")

    if export:
        if not os.path.exists("images/"):
            os.mkdir("images/")
        pyplot.savefig("images/2D_GMM_demonstration.png", dpi=100, format="png")

    pyplot.show()
