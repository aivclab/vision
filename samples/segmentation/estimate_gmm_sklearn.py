import numpy
from sklearn.mixture import GaussianMixture

from neodroidvision.segmentation.gmm import visualization

if __name__ == "__main__":

    N, D = 1000, 3

    if D == 2:

        means = numpy.array([[0.5, 0.0], [0, 0], [-0.5, -0.5], [-0.8, 0.3]])
        covs = numpy.array(
            [
                numpy.diag([0.01, 0.01]),
                numpy.diag([0.025, 0.01]),
                numpy.diag([0.01, 0.025]),
                numpy.diag([0.01, 0.01]),
            ]
        )
    elif D == 3:

        means = numpy.array(
            [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.5, -0.5, -0.5], [-0.8, 0.3, 0.4]]
        )
        covs = numpy.array(
            [
                numpy.diag([0.01, 0.01, 0.03]),
                numpy.diag([0.08, 0.01, 0.01]),
                numpy.diag([0.01, 0.05, 0.01]),
                numpy.diag([0.03, 0.07, 0.01]),
            ]
        )

    n_components = means.shape[0]

    points = []
    for i in range(len(means)):
        x = numpy.random.multivariate_normal(means[i], covs[i], N)
        points.append(x)
    points = numpy.concatenate(points)

    gmm = GaussianMixture(n_components=n_components, covariance_type="diag")
    gmm.fit(points)

    if D == 2:
        visualization.visualise_2D_gmm(
            points, gmm.weights_, gmm.means_.T, numpy.sqrt(gmm.covariances_).T
        )
    elif D == 3:
        visualization.visualise_3d_gmm(
            points, gmm.weights_, gmm.means_.T, numpy.sqrt(gmm.covariances_).T
        )
