#!/usr/bin/env python
# -*- coding: utf-8 -*-
# registration.py
"""
Classes for registering point sets.

Based on:
- Myronenko and Xubo Song - 2010 - Point Set Registration Coherent Point Drift
DOI: 10.1109/TPAMI.2010.46

Copyright (c) 2018, David Hoffman
"""

import itertools
import numpy as np
import scipy.spatial as spatial
import scipy.spatial.distance as distance
import scipy.linalg as la
from skimage.transform._geometric import _umeyama

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# get a logger
import logging

logger = logging.getLogger(__name__)


class BaseCPD(object):
    """Base class for the coherent point drift algorithm.
    
    Based on:
    Myronenko and Xubo Song - 2010 - Point Set Registration Coherent Point Drift
    DOI: 10.1109/TPAMI.2010.46
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """Set up the registration class that will actually perform the CPD algorithm.
        
        Parameters
        ----------
        X : ndarray (N, D)
            Fixed point cloud, an N by D array of N original observations in an n-dimensional space
        Y : ndarray (M, D)
            Moving point cloud, an M by D array of N original observations in an n-dimensional space
        """
        # save an internal copy so make sure nothings being mucked with
        self.X = self._X = X
        self.Y = self._Y = Y
        # extract the dimensions
        self.N, self.D = self.X.shape
        self.M, D = self.Y.shape
        assert D == self.D, "Point clouds have different dimensions"
        assert self.N, "X has no points"
        assert self.M, "Y has no points"

    def __repr__(self) -> str:
        """Nice representation of the model."""
        basestr = "Model = {}, X = {},  Y = {}".format(self.__class__, self.X.shape, self.Y.shape)
        try:
            extrastr = ", w = {}, i = {}, Q = {}, B = {}, t = {}".format(
                self.w, self.iteration, self.Q, self.B, self.translation
            )
        except AttributeError:
            extrastr = ", registration not run"
        return basestr + extrastr

    @property
    def scale(self):
        """Return the estimated scale of the transformation matrix"""
        return self.B.mean(axis=1)

    @property
    def matches(self) -> np.ndarray:
        """Return X, Y matches."""
        return np.where(self.p_old > max(min(self.w, 0.9), np.finfo(float).eps))[::-1]

    def _estimate(self):
        """Overload in the child classes."""
        raise NotImplementedError

    def estimate(self):
        """Estimate the simple transform for matching pairs."""
        logger.debug(
            "Doing a simple estimation of the transformation for {}".format(self.__class__)
        )
        self._estimate()
        # this assumes it's being called from a child class
        self.updateTY()
        # the match matrix is just the identity matrix by definition
        self.p_old = np.eye(self.N, self.M)
        # these need to be filled in for the ploting and str function to work
        self.iteration = "N/A"
        self.w = 0
        self.Q = "N/A"

    def plot(self, only2d=False):
        """Plot the results of the registration."""
        if self.X.shape[-1] > 1:
            if self.X.shape[-1] > 2 and not only2d:
                projection = "3d"
                s = slice(None, 3)
            else:
                projection = None
                s = slice(None, 2)

            fig = plt.figure(figsize=(8, 4))
            ax0 = fig.add_subplot(121, projection=projection)
            ax1 = fig.add_subplot(122)
            axs = (ax0, ax1)

            ax0.scatter(*self.Y.T[s], marker=".", c="g")
            ax0.scatter(*self.TY.T[s], marker="o", c="b")
            ax0.scatter(*self.X.T[s], marker="x", c="r")
            ax0.quiver(*self.Y.T[s], *(self.TY.T[s] - self.Y.T[s]), color="orange", pivot="tail")
            if projection is None:
                ax0.set_aspect("equal")
            ax0.set_title(
                "RMSE = {:.3f}, i = {}\ntvec = {}".format(
                    self.rmse, self.iteration, self.translation
                )
            )
        else:
            fig, ax1 = plt.subplots(1)
            axs = (ax1,)
        ax1.matshow(self.p_old)

        ax1.set_aspect("auto")
        ax1.set_title(
            "Num pnts = {}, numcorr = {}".format(len(self.TY), (self.p_old > self.w).sum())
        )
        return fig, axs

    def transform(self, other: np.ndarray) -> np.ndarray:
        """Transform `other` point cloud via the Y -> X registration."""
        return other @ self.B.T + self.translation

    def updateTY(self):
        """Update the transformed point cloud and distance matrix."""
        self.TY = self.transform(self.Y)
        # we need to update the distance matrix too
        # This gives us a matrix of ||x - T(y)||**2, eq (1)
        # But we want the rows to be m and columns n
        self.dist_matrix = distance.cdist(self.TY, self.X, "sqeuclidean")
        # make sure we have the right shape
        assert self.dist_matrix.shape == (self.M, self.N), "Error with dist_matrix"

    # these are defined in the paper but not used, included here for completeness
    # @property
    # def pGMM(self):
    #     """The probability density of the gaussian mixture model along the fixed points"""
    #     norm_factor = self.M * (2 * np.pi * self.var) ** (self.D / 2)
    #     p_mat = np.exp(- self.dist_matrix / 2 / self.var) / norm_factor
    #     # sum along the fixed points
    #     return p_mat.sum(0)

    # @property
    # def p(self):
    #     """The total probability including the uniform distribution"""
    #     return self.w / self.N + (1 - self.w) * self.pGMM

    def estep(self):
        """Do expectation step were we calculate the posterior probability of the GMM centroids."""
        # calculate "P_old" via equation 6
        p_mat = np.exp(-self.dist_matrix / 2 / self.var)
        c = (2 * np.pi * self.var) ** (self.D / 2)
        c *= self.w / (1 - self.w)
        c *= self.M / self.N
        # sum along the moving points, i.e. along M
        denominator = p_mat.sum(0, keepdims=True)
        assert denominator.shape == (1, self.N), "Calculation of denominator failed {}".format(
            denominator.shape
        )
        # check if denominator is all zeros, which means p_mat is all zeros
        if (denominator <= np.finfo(float).eps).all():
            # then the final p should just be a uniform distribution
            # should log or warn user this is happening
            logger.debug("P_mat is null, resetting to uniform probabilities")
            p_old = np.ones_like(p_mat) / self.M
        else:
            if c < np.finfo(float).eps:
                logger.debug("c is small, setting to eps")
                c = np.finfo(float).eps
            p_old = p_mat / (denominator + c)
        # compute Np, make sure it's neither zero nor more than N
        self.Np = min(self.N, max(p_old.sum(), np.finfo(float).eps))
        # update Q so we can track convergence using equation (5)
        self.Q = (p_old * self.dist_matrix).sum() / 2 / self.var + self.Np * self.D * np.log(
            self.var
        ) / 2
        # store p_old
        self.p_old = p_old

    def updateB(self):
        """Update B matrix.
        
        This is the only method that needs to be overloaded for the various linear transformation subclasses,
        more will need to be done for non-rigid transformation models.
        """
        raise NotImplementedError

    def mstep(self):
        """Maximization step.
        
        Update transformation and variance these are the transposes of the equations on p. 2265 and 2266
        """
        # calculate intermediate values
        mu_x = (self.p_old @ self.X).sum(0, keepdims=True) / self.Np
        mu_y = (self.p_old.T @ self.Y).sum(0, keepdims=True) / self.Np
        assert mu_x.size == mu_y.size == self.D, "Dimensions on mu's are wrong"
        Xhat = self.Xhat = self.X - mu_x
        Yhat = self.Yhat = self.Y - mu_y

        # calculate A
        self.A = A = Xhat.T @ self.p_old.T @ Yhat

        # calculate B
        B = self.updateB()

        # calculate translation
        self.translation = mu_x - mu_y @ B.T

        # calculate estimate of variance
        self.var = np.trace(Xhat.T @ np.diag(self.p_old.sum(0)) @ Xhat) - np.trace(A @ B.T)
        self.var /= self.Np * self.D
        logger.debug("Variance is {}".format(self.var))
        # make sure self.var is positive
        if self.var < np.finfo(float).eps:
            # self.var = np.finfo(float).eps
            self.var = self.tol
            logger.warning(
                "Variance has dropped below machine precision, setting to {}".format(self.var)
            )
            # self.var = self.init_var = self.init_var * 2
            # self.translation = -self.Y.mean(axis=0) + self.X.mean(axis=0)
            # print("Var small resetting to", self.var)

    def calc_var(self):
        """Calculate variance in transform."""
        return self.dist_matrix.sum() / (self.D * self.N * self.M)

    @property
    def rmse(self):
        """Return RMSE between X and transformed Y."""
        # need to weight the RMSE by the probability matrix ...
        return np.sqrt((self.p_old * self.dist_matrix).mean())
        # return np.sqrt(((self.X - self.TY)**2).sum(1)).mean()

    def calc_init_scale(self):
        """Need to overloaded in child classes."""
        raise NotImplementedError

    def norm_data(self):
        """Normalize data to mean 0 and unit variance."""
        # calculate mean displacement
        logger.debug("Normalizing data")
        self.ty = self.Y.mean(0, keepdims=True)
        self.tx = self.X.mean(0, keepdims=True)
        logger.debug("tx = {}, ty = {}".format(self.tx, self.ty))
        # move point clouds
        self._Y_orig = self.Y
        self._X_orig = self.X
        self.Y = self.Y - self.ty
        self.X = self.X - self.tx
        # calculate scale
        self.calc_init_scale()
        logger.debug("scale_x = {}, scale_y = {}".format(self.scale_x, self.scale_y))
        # apply scale
        Sx = np.diag(self.scale_x)
        Sy = np.diag(self.scale_y)
        Sx_1 = np.diag(1 / self.scale_x)
        Sy_1 = np.diag(1 / self.scale_y)
        self.Y = self.Y @ Sy
        self.X = self.X @ Sx

        self.translation = (self.ty @ self.B.T + self.translation - self.tx) @ Sx
        logger.debug(f"B = {self.B}")
        self.B = Sx @ self.B @ Sy_1
        logger.debug(f"B = {self.B}")

    def unnorm_data(self):
        """Undo the intial normalization."""
        logger.debug("Undoing normalization")
        logger.debug("tx = {}, ty = {}".format(self.tx, self.ty))
        logger.debug("scale_x = {}, scale_y = {}".format(self.scale_x, self.scale_y))
        Sx = np.diag(self.scale_x)
        Sy = np.diag(self.scale_y)
        Sx_1 = np.diag(1 / self.scale_x)
        Sy_1 = np.diag(1 / self.scale_y)
        # the scale matrices are diagonal so S.T == S
        self.Y = self.Y @ Sy_1 + self.ty
        self.X = self.X @ Sx_1 + self.tx
        assert np.allclose(self.Y, self._Y_orig), "Failed to revert"
        assert np.allclose(self.X, self._X_orig), "Failed to revert"
        # B doesn't need to be transposed and
        self.B = Sx_1 @ self.B @ Sy
        self.translation = -self.ty @ self.B.T + self.translation @ Sx_1 + self.tx

    def __call__(
        self, tol=1e-6, dist_tol=0, maxiters=1000, init_var=None, weight=0, normalization=True
    ):
        """Perform the actual registration.

        Parameters
        ----------
        tol : float
        dist_tol : float
            Stop the iteration of the average distance between matching points is
            less than this number. This is really only necessary for synthetic data
            with no noise
        maxiters : int
        init_var : float
        weight : float
        B : ndarray (D, D)
        translation : ndarray (1, D)
        """
        # initialize transform
        self.translation = np.ones((1, self.D))
        self.B = np.eye(self.D)
        self.tol = tol

        # update to the initial position
        if normalization:
            self.norm_data()
        self.updateTY()

        # set up initial variance
        if init_var is None:
            init_var = self.calc_var()
        self.var = self.init_var = init_var
        logger.debug("self.init_var = {}".format(self.var))

        # initialize the weight of the uniform distribution
        assert 0 <= weight < 1, "Weight must be between 0 and 1"
        self.w = weight

        for self.iteration in range(maxiters):
            # do iterations expectation, maximization followed by transformation
            self.estep()
            self.mstep()
            self.updateTY()
            if self.iteration > 0:
                # now update Q to follow convergence
                # we want to minimize Q so Q_old should be more positive than the new Q
                Q_delta = np.abs(self.Q_old - self.Q)  # / np.abs(self.Q_old)
                if Q_delta < 0:
                    logger.warning("Q_delta = {}".format(Q_delta))
                logger.debug("Q_delta = {}".format(Q_delta))
                if Q_delta <= tol:
                    logger.info("Objective function converged, Q_delta = {:.3e}".format(Q_delta))
                    break
                if self.rmse <= dist_tol:
                    logger.info("Average distance converged")
                    break
            self.Q_old = self.Q
        else:
            logger.warning(
                (
                    "Maximum iterations ({}) reached without" + " convergence, final Q = {:.3e}"
                ).format(self.iteration, self.Q)
            )
        # update p matrix once more
        self.estep()
        # unnorm the data and apply the final transformation.
        if normalization:
            self.unnorm_data()
        self.updateTY()

        return self.TY


class TranslationCPD(BaseCPD):
    """Coherent point drift with a translation only transformation model."""

    def updateB(self):
        """Update step.
        
        Translation only means that B should be identity.
        """
        self.B = np.eye(self.D)
        return self.B

    def calc_init_scale(self):
        """For translation only we need to calculate a uniform scaling."""
        anisotropic_scale = np.concatenate((self.X, self.Y)).std(0)
        self.scale_x = self.scale_y = 1 / anisotropic_scale

    def _estimate(self):
        """Estimate the translation transform."""
        self.B, self.translation = np.eye(self.D), (self.X - self.Y).mean(0, keepdims=True)


class SimilarityCPD(BaseCPD):
    """Coherent point drift with a similarity (translation, rotation and isotropic scaling) transformation model."""

    # this class is specifically designed so that it can be easily subclassed to represent
    # a rigid transformation model.
    def calculateR(self):
        """Calculate the estimated rotation matrix, eq. (9)."""
        U, S, VT = la.svd(self.A)
        c = np.ones_like(S)
        c[-1] = la.det(U @ VT)
        C = np.diag(c)
        self.R = U @ C @ VT
        return self.R

    def calculateS(self):
        """Calculate the scale factor, Fig 2 p. 2266."""
        a = self.Yhat.T @ np.diag(self.p_old.sum(1)) @ self.Yhat
        self.s = np.trace(self.A.T @ self.R) / np.trace(a)
        return self.s

    def updateB(self):
        """Update B: in this case is just the rotation matrix multiplied by the scale factor."""
        R = self.calculateR()
        s = self.calculateS()
        self.B = s * R
        return self.B

    def calc_init_scale(self):
        """Calculate scale: for similarity we have isotropic scaling for each point cloud."""
        # we can prescale by the same anisotropic scaling factor we use in
        # TranslationCPD and then augment it by an isotropic scaling factor
        # for each point cloud.
        anisotropic_scale = np.concatenate((self.X, self.Y)).std()
        # self.scale_x = anisotropic_scale / self.X.var()
        # self.scale_y = anisotropic_scale / self.Y.var()
        # NOTE: the above doesn't work
        self.scale_x = self.scale_y = 1 / np.array((anisotropic_scale, anisotropic_scale))

    def _umeyama(self):
        """Calculate Umeyama: for similarity we want to have scaling."""
        # the call signature for _umeyama is (src, dst)
        # which is the reverse of ours
        return _umeyama(self.Y, self.X, True)

    def _estimate(self):
        """Estimate the similarity transform."""
        T = self._umeyama()
        D = self.D
        # T is in the usual orientation
        B = T[:D, :D]
        translation = T[:D, -1:].T
        assert np.allclose(
            T[-1, :], np.concatenate((np.zeros(D), np.ones(1)))
        ), "Error, T = {}".format(T)
        self.B, self.translation = B, translation


class RigidCPD(SimilarityCPD):
    """Coherent point drift with a rigid or Euclidean (translation and rotation) transformation model."""

    def calculateS(self):
        """No scaling for this guy."""
        return 1

    def _umeyama(self):
        """For this class we want to have _umeyama without scaling."""
        # the call signature for _umeyama is (src, dst)
        # which is the reverse of ours
        return _umeyama(self.Y, self.X, False)

    # for rigid we also want to avoid anything other than uniform scaling
    calc_init_scale = TranslationCPD.calc_init_scale


EuclideanCPD = RigidCPD


class AffineCPD(BaseCPD):
    """Coherent point drift with a similarity (translation, rotation, shear and anisotropic scaling) transformation model."""

    def updateB(self):
        """Solve for B using equations in Fig. 3 p. 2266."""
        a = self.Yhat.T @ np.diag(self.p_old.sum(1)) @ self.Yhat
        # solve B = self.A @ np.inv(a) == B @ a = self.A == a.T @ B.T = self.A.T
        # self.B = la.solve(a.T, self.A.T).T
        # a is a symmetric matrix
        self.B = la.solve(a, self.A.T).T
        return self.B

    def calc_init_scale(self):
        """Calculate scale."""
        # For affine we have anisotropic scaling for each point cloud along each dimension
        self.scale_x = 1 / self.X.std(0)
        self.scale_y = 1 / self.Y.std(0)

    def _estimate(self):
        """Estimate the affine transformation for a set of corresponding points."""
        # affine is quite simple, we want to solve the equation A @ Y = X
        # or Y.T @ A.T = X.T
        # where Y and X are augmented matrices (an extra row of ones)
        # https://en.wikipedia.org/wiki/Affine_transformation#Augmented_matrix
        aug_X = np.hstack((self.X, np.ones((self.N, 1))))
        aug_Y = np.hstack((self.Y, np.ones((self.N, 1))))
        # pull the dimension out
        D = self.D
        # solve for matrix transforming Y to X
        T, res, rank, s = la.lstsq(aug_Y, aug_X)
        # remember that B = A.T (A not augmented)
        B = T[:D, :D].T
        # we want to keep the extra dimension for translation
        translation = T[-1:, :D]
        # make sure that the solution makes sense (last column should be 1 | 0)
        assert np.allclose(
            T[:, -1], np.concatenate((np.zeros(D), np.ones(1)))
        ), "Error\nT = {}\nX = {}\nY = {}".format(T, self.X, self.Y)
        self.B, self.translation = B, translation


# a dictionary to choose models from
model_dict = {
    "translation": TranslationCPD,
    "rigid": RigidCPD,
    "euclidean": EuclideanCPD,
    "similarity": SimilarityCPD,
    "affine": AffineCPD,
}


def choose_model(model):
    """Choose model if string."""
    if isinstance(model, str):
        model = model_dict[model.lower()]
    elif not issubclass(model, BaseCPD):
        raise ValueError("Model {} is not recognized".format(model))
    return model


def auto_weight(X, Y, model, resolution=0.01, limits=0.05, **kwargs):
    """Automatically determine the weight to use in the CPD algorithm.

    Parameters
    ----------
    X : ndarray (N, D)
        Fixed point cloud, an N by D array of N original observations in an n-dimensional space
    Y : ndarray (M, D)
        Moving point cloud, an M by D array of N original observations in an n-dimensional space
    model : str or BaseCPD child class
        The transformation model to use, available types are:
            Translation
            Rigid
            Euclidean
            Similarity
            Affine
    resolution : float
        the resolution at which to sample the weights
    limits : float or length 2 iterable
        The limits of weight to search
    kwargs : dictionary
        key word arguments to pass to the model function when its called.

    """
    # test inputs
    model = choose_model(model)

    try:
        # the user has passed low and high limits
        limit_low, limit_high = limits
    except TypeError:
        # the user has passed a single limit
        limit_low = limits
        limit_high = 1 - limits
    # generate weights to test
    ws = np.arange(limit_low, limit_high, resolution)
    # container for various registrations
    regs = []
    # iterate through weights
    for w in ws:
        kwargs.update(weight=w)
        reg = model(X, Y)
        reg(**kwargs)
        regs.append(reg)

    # if the dimension of the data is less than 3 use the 1 norm
    # else use the frobenius norm. This is a heuristic based on simulated data.
    if reg.D < 3:
        norm_type = 1
    else:
        norm_type = "fro"
    # look at all the norms of the match matrices (The match matrix should be sparse
    # and the norms we've chosen maximize sparsity)
    norm = np.asarray([np.linalg.norm(reg.p_old, norm_type) for reg in regs])
    # find the weight that maximizes sparsity
    w = ws[norm.argmax()]
    # update and run the model
    kwargs.update(weight=w)
    reg = model(X, Y)
    reg(**kwargs)
    # return the model to the user
    return reg


def nearest_neighbors(fids0, fids1, r=100, transform=lambda x: x, coords=["x0", "y0"]):
    """Find nearest neighbors in both sets."""
    idx00, idx01 = closest_point_matches(
        fids0[coords].values, transform(fids1[coords].values), r=r
    )
    idx11, idx10 = closest_point_matches(
        transform(fids1[coords].values), fids0[coords].values, r=r
    )
    fids0_filt = fids0.iloc[idx10]
    fids1_filt = fids1.iloc[idx01]
    idx0, idx1 = closest_point_matches(
        fids0_filt[coords].values, transform(fids1_filt[coords].values), r=r
    )
    return fids0_filt.iloc[idx0], fids1_filt.iloc[idx1]


# def normed_rmse(reg):
#     reg.norm_data()
#     reg.updateTY()
#     rmse = reg.rmse
#     reg.unnorm_data()
#     reg.updateTY()
#     return rmse


def align(
    fids0,
    fids1,
    atol=1,
    rtol=1e-3,
    diagnostics=False,
    model="translation",
    only2d=False,
    iters=100,
):
    """Align two slabs fiducials, assumes that z coordinate has been normalized."""
    model = choose_model(model)

    def register(fids0_filt, fids1_filt, coords):
        reg = model(fids0_filt[coords].values, fids1_filt[coords].values)
        try:
            reg.estimate()
        except ValueError:
            reg(weight=0.05)
        return reg

    def sub_func(rmse, transform, coords):
        for i in range(iters):
            r = max(rmse * 2, 1)
            try:
                fids0_filt, fids1_filt = nearest_neighbors(
                    fids0, fids1, r=r, transform=transform, coords=coords
                )
            except ValueError:
                rmse *= 2
                continue
            reg = register(fids0_filt, fids1_filt, coords)
            transform = reg.transform
            rmse_new = reg.rmse
            rmse_rel = (rmse - rmse_new) / rmse
            if rmse_new < atol or rmse_rel < rtol:
                break
            rmse = rmse_new
        else:
            logger.error(
                "{} failed, rmse = {}, rel = {}, i = {}".format(coords, rmse_new, rmse_rel, i)
            )

        logger.info(
            "{} succeeded, rmse = {}, rel = {}, i = {}".format(coords, rmse_new, rmse_rel, i)
        )

        if diagnostics:
            reg.plot()
        return reg, fids0_filt, fids1_filt

    reg2d, fids0_filt, fids1_filt = sub_func(50, lambda x: x, ["x0", "y0"])

    if only2d:
        return reg2d

    new_transform = register(fids0_filt, fids1_filt, ["x0", "y0", "z0"]).transform
    reg3d, _, _ = sub_func(reg2d.rmse, new_transform, ["x0", "y0", "z0"])

    return reg3d


def closest_point_matches(X, Y, method="tree", **kwargs):
    """Keep determine the nearest neighbors in two point clouds.

    Parameters
    ----------
    X : ndarray (N, D)
    Y : ndarray (M, D)

    kwargs
    ------
    r : float
        The search radius for nearest neighbors

    Returns
    -------
    xpoints : ndarray
        indicies of points with neighbors in x
    ypoints : ndarray
        indicies of points with neighbors in y
    """
    if method.lower() == "tree":
        return _keepclosesttree(X, Y, **kwargs)
    elif method.lower() == "brute":
        return _keepclosestbrute(X, Y, **kwargs)
    else:
        raise ValueError("Method {} not recognized".format(method))


def _keepclosestbrute(X, Y, r=10, percentile=None):
    # calculate the distance matrix
    dist_matrix = distance.cdist(X, Y)
    # if user requests percentile
    if percentile is not None:
        r = np.percentile(dist_matrix, percentile * (len(X) + len(Y)) / (len(X) * len(Y)))
    logger.debug(
        "r = {}, fraction pairs kept = {}".format(r, (dist_matrix < r).sum() / dist_matrix.size)
    )
    result = [np.unique(a) for a in np.where(dist_matrix < r)]
    # log percentages
    logger.debug(
        "percentage x kept = {}, y kept = {}".format(
            *[len(a) / len(aa) for a, aa in zip(result, (X, Y))]
        )
    )
    return result


def _keepclosesttree(X, Y, r=10):
    # build the trees
    ytree = spatial.cKDTree(Y)
    # find nearest neighbors to each point in X in Y, ii are the indices of ytree that match X
    dd, ii = ytree.query(X)
    # filter
    filt = dd < r
    xpoints = np.arange(len(X))[filt]
    ypoints = ii[filt]
    # check for duplicate indexs
    uypoints = np.unique(ypoints)
    uxpoints = np.unique(xpoints)
    if uxpoints.size < xpoints.size or uypoints.size < ypoints.size:
        logger.debug("taking unique points")
        xpoints, ypoints = uxpoints, uypoints
    logger.debug(
        "percentage x kept = {}, y kept = {}".format(len(xpoints) / len(X), len(ypoints) / len(Y))
    )

    return xpoints, ypoints


# These are tools to propogate slab to slab alignment
def to_augmented(B, t):
    """Convert transform matrix and translation vector to an augmented transformation matrix.

    https://en.wikipedia.org/wiki/Affine_transformation#Augmented_matrix
    """
    # get the dimension of the transformation
    d = len(B)
    # build augmented matrix
    aug_B = np.zeros((d + 1, d + 1))
    aug_B[:d, :d] = B
    aug_B[-1, :d] = t
    aug_B[-1, -1] = 1
    return aug_B


def from_augmented(aug_B):
    """Convert the augmented matrix back into transformation matrix + tranlsation vector."""
    return aug_B[:-1, :-1], aug_B[-1, :-1]


def propogate_transforms(regs, initial=None):
    """Propagate transforms along slabs."""
    # initialize
    if initial is None:
        initial = np.eye(regs[0].D + 1)
    aug_Bs = [initial]
    # make list of all augmented matrices
    for reg in regs:
        aug_Bs.append(to_augmented(reg.B, reg.translation))
    # initialize Bs and ts
    Bs = []
    ts = []
    # propogate
    # for list of augmented matrices M0, M1, M2, ...
    # return M0, M0 @ M1, M0 @ M1 @ M2, ...
    for aug_B2 in itertools.accumulate(aug_Bs, lambda a, b: b @ a):
        # split the propogated transform and save each one
        B, t = from_augmented(aug_B2)
        Bs.append(B)
        ts.append(t)

    return Bs, ts


def apply_transform_to_slab(s, B, t, copy=True):
    """Apply a given transform to a slab."""
    coords = ["x0", "y0", "z0"][: len(B)]
    # copy slab so as to not change original.
    if copy:
        s2 = s.copy()
    else:
        s2 = s
    s2[coords] = s[coords].values @ B.T + t
    return s2
