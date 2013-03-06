## This code is written by Davide Albanese <davide.albanese@gmail.com>
## Copyright (C) 2013 Davide Albanese

## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from scipy.spatial.distance import pdist, squareform


def _compute_AB(x, y, index):
    xa = np.atleast_2d(x)
    ya = np.atleast_2d(y)       

    if xa.ndim > 2 or ya.ndim > 2:
        raise ValueError("x and y must be 1d or 2d array_like objects")

    if xa.shape[0] == 1:
        xa = xa.T

    if ya.shape[0] == 1: 
        ya = ya.T

    if xa.shape[0] != ya.shape[0]:
        raise ValueError("x and y must have the same sample sizes")
       
    if index <= 0 or index > 2:
        raise ValueError("index must be in (0, 2]")

    # compute A
    a_kl = squareform(pdist(xa, 'euclidean')**index)
    a_k = np.mean(a_kl, axis=1).reshape(-1, 1)
    a_l = a_k.T
    a = np.mean(a_kl)
    A = a_kl - a_k - a_l + a

    # compute B
    b_kl = squareform(pdist(ya, 'euclidean')**index)
    b_k = np.mean(b_kl, axis=1).reshape(-1, 1)
    b_l = b_k.T
    b = np.mean(b_kl)
    B = b_kl - b_k - b_l + b

    return A, B


def dcor_complete(x, y, index=1.0):
    """Distance Correlation and Covariance Statistics.

    Computes distance covariance and distance correlation 
    statistics, which are multivariate measures of dependence.
    Distance correlation is a measure of dependence between 
    random vectors introduced by Szekely, Rizzo, and Bakirov (2007).
    
    For all distributions with finite first moments, distance 
    correlation R generalizes the idea of correlation in two 
    fundamental ways: (1) R(X, Y) is defined for X and Y
    in arbitrary dimension. (2) R(X, Y) = 0 characterizes 
    independence of X and Y. The sample sizes of the two samples 
    must agree.

    Distance correlation satisfies 0 <= R <= 1, and R = 0 only if 
    X and Y are independent.

    :Parameters:
        x : 1d (N) or 2d (N,P) array_like object
            data of first sample
        y : 1d (N) or 2d (N,Q) array_like objects
            data of second sample
        index : float (0,2]
            index exponent on Euclidean distance

    :Returns:
        dCov, dCor, dVarX, dVarY : tuple of floats
            sample distance covariance, sample distance 
            correlation, distance variance of x sample,
            distance variance of y sample
            
    Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007), 
    Measuring and Testing Dependence by Correlation of Distances, 
    Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.
    """

    EPS = np.finfo(np.float).eps
    
    # Definition 4 page 2773
    
    A, B = _compute_AB(x, y, index)
    
    # eq. 2.8, 2.9
    dcov_2 = np.mean(A * B)
    dvarx_2 = np.mean(A**2)
    dvary_2 = np.mean(B**2)

    # eq. 2.10
    dvarxy_2 = dvarx_2 * dvary_2
    if dvarxy_2 > EPS:
        dcor_2 = dcov_2 / np.sqrt(dvarxy_2)
    else:
        dcor_2 = 0.0
        
    dcov, dcor, dvarx, dvary = np.sqrt(dcov_2), np.sqrt(dcor_2), \
        np.sqrt(dvarx_2), np.sqrt(dvary_2)
    
    return dcov, dcor, dvarx, dvary


def dcov(x, y, index=1.0):
    """Distance Covariance.

    See the `dcor_complete()` documentation.

    :Parameters:
        x : 1d (N) or 2d (N,P) array_like object
            data of first sample
        y : 1d (N) or 2d (N,Q) array_like objects
            data of second sample
        index : float (0,2]
            index exponent on Euclidean distance

    :Returns:
        dCov : float
            sample distance covariance    
    """

    A, B = _compute_AB(x, y, index)
    dcov_2 = np.mean(A * B)
    return np.sqrt(dcov_2)


def dcor(x, y, index=1.0):
    """Distance Correlation.

    See the `dcor_complete()` documentation.

    :Parameters:
        x : 1d (N) or 2d (N,P) array_like object
            data of first sample
        y : 1d (N) or 2d (N,Q) array_like objects
            data of second sample
        index : float (0,2]
            index exponent on Euclidean distance

    :Returns:
        dCor : float
            sample distance correlation    
    """

    _, dcor, _, _ = dcor_complete(x, y, index=1.0)
    return dcor


