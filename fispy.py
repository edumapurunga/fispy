# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 22:49:28 2020

This module presents a set of algorithms to calculate or approximate
the fisher information matrix for a given system.

@author: edumapurunga
"""
# Define all functions
__all__ = ['sym2lti', 'matsim', 'psiji', 'psijil', 'fishinf', 'covdata', 'arma_acorr', 'arma_ccorr']

#%% Necessary libraries
import numpy as np
import sympy as sym
import scipy.signal as sg
#%% Functions
# Convert Sympy to LTI
def sym2lti(xpr, s=sym.Symbol('z'), t=None):
    """ Convert Sympy transfer function polynomial to Scipy LTI """
    num, den = sym.simplify(xpr).as_numer_denom()  # expressions
    p_num_den = sym.poly(num, s), sym.poly(den, s)  # polynomials
    #c_num_den = [sym.expand(p).all_coeffs() for p in p_num_den]  # coefficients
    c_num_den = [p.all_coeffs() for p in p_num_den]
    # If there is no parameter
    if t == None:
        l_num, l_den = [sym.lambdify((), c)() for c in c_num_den]  # convert to floats
    else:
        l_num, l_den = c_num_den
    # Put the zeros on the numerator to be compatible with lfilter
    # Numerator order
    m = len(l_num) - 1
    nk = 0
    for i in l_den[::-1]:
        if i == 0:
            nk += 1
            #Removing the delays
            l_den.pop()
    # Denominator order
    n = len(l_den) - 1
    # pole excess
    d = n-m
    l_num = [0]*(nk+d) + l_num
    # return (l_num, l_den)
    return (np.array(l_num), np.array(l_den))

# Simulate Matrix based transfer functions
def matsim(M, r):
    """
    This functions simulates a transfer matrix of discrete-time invariant
    linear as:
        w(q) = M(q)r(t)

    Parameters
    ----------
    M : list of tuples
        The transfer matrix to be simulated. Each entry must contain a tuple
        corresponding to the numerator and denominator.
    r : list of lists
        The inputs to be used in the matrix .

    Returns
    -------
    w : TYPE
        DESCRIPTION.

    """
    ni = len(r[0])
    N = len(r)
    nr = len(M)
    nc = len(M[0])
    w = [[] for i in range(ni)]
    for i in range(0, nr):
        for j in range(0, nc):
            w[i].append(sg.lfilter(M[i][j][0], M[i][j][1], r[:, j:j+1], axis=0))
    return w

# Get the derivatives
def psiji(T, j, i, var, D=dict()):
    """
    Get the derivative with respect to the parameters for the transfer function
    matrix T

    Parameters
    ----------
    T : list or sympy matrix
        Matrix of input-output representation.
    j : int
        selected output.
    i : int
        selected input.
    var : list
        unknown sympy variables.

    Returns
    -------
    a numpy function
        the derivative with respect to the parameters.

    """
    # Number of Unknowns
    d = len(var)
    return [sym2lti(sym.simplify(sym.diff(T[j, i], var[k]).subs(D)), t=1) for k in range(d)]

def psijil(T, j, i, var):
    """
    Get the derivative with respect to the parameters for the transfer function
    matrix T

    Parameters
    ----------
    T : list or sympy matrix
        Matrix of input-output representation.
    j : int
        selected output.
    i : int
        selected input.
    var : list
        unknown sympy variables.

    Returns
    -------
    a numpy function
        the derivative with respect to the parameters.

    """
    # Number of Unknowns
    d = len(var)
    return sym.lambdify(var, [sym2lti(sym.simplify(sym.diff(T[j, i], var[k])), t=1) for k in range(d)], "numpy")

# Get the symbolic covariance
def fishinf(var, mpsi, ne, nm, R, Q):
    # Number of variables
    d = len(var)
    # 
    M = sym.zeros(d, d)
    for j in range(len(nm)):
        for i in range(len(ne)):
            # Auxiliary M
            Maux = sym.zeros(d, d)
            # Get the correct psi
            psi = mpsi[nm[j]][ne[i]]
            for k1 in range(d):
                for k2 in range(d):
                    if k1 == k2:
                        if np.count_nonzero(psi[k1][0]):
                            Maux[k1, k1] = arma_acorr(psi[k1][1], psi[k1][0], R[ne[i]], 1)[0][1]
                    elif k2 > k1:
                        if np.count_nonzero(psi[k1][0]) and np.count_nonzero(psi[k2][0]):
                            Maux[k1, k2] = arma_ccorr(psi[k1][1], psi[k1][0], psi[k2][1], psi[k2][0], R[ne[i]], 1)[0][1]
                    else:
                        Maux[k1, k2] = Maux[k2, k1]
            # Divide by the correct Q
            Maux = Maux/Q[nm[j], nm[j]]
            M += Maux
    return sym.simplify(M)

# Get the covariance
def covdata(to, mpsi, ne, nm, u, y, Q):
    """
    Returns an estimate of the theoretical covariance matrix of pem for
    loop systems

    Parameters
    ----------
    orders : Tuple of tuples
        A tuple containing the orders of each transfer function (na, nb, nk).
    ne : list
        List of nodes that are excited.
    nm : list
        List of nodes that are measured.
    u : numpy.ndarray
        Inputs.
    y : numpy.ndarray
        Outputs.

    Returns
    -------
    P : numpy.ndarray
        Estimate of the covariance matrix.

    """
    # Number of Data
    Nu, nu = np.shape(u)
    Ny, ny = np.shape(y)
    N = Ny
    # Number of parameters
    d = len(to)
    # Initialize the derivatives
    M = np.zeros((d, d))
    # Iterate over the possible combinations
    for j in range(len(nm)):
        for i in range(len(ne)):
                # Store
                psitf = mpsi[nm[j]][ne[i]](*to)
                psi = np.zeros((d, N))
                for k in range(d):
                    psi[k, :] = sg.lfilter(psitf[k][0], psitf[k][1], u[:, i], axis=0)
                M += psi.dot(psi.T)/Q[j, j]
    return np.linalg.inv(M/N)

# Build Toeplitz Matrix
def toeplitz(r):
    # c and r must be Matrix objects
    T = sym.Matrix([])
    N = r.shape[1]
    for j in range(N):
        if j == 0:
            T = T.row_insert(j, r)
        else:
            T = T.row_insert(j, sym.Matrix([[sym.zeros(1, j), r[0, :N-j]]]))
    return T

# Build hankel matrix
def hankel(c):
    H = sym.Matrix([])
    for j in range(c.shape[0]):
        if j == 0:
            H = H.col_insert(j, c)
        else:
            H = H.col_insert(j, sym.Matrix([[c[j:, 0]], [sym.zeros(j, 1)]]))
    return H
    
# Arma acorr
def arma_acorr(A, C, var, maxlag):
    # Description to help the user
    """Function that calculates the theoretical autocorrelation function of an ARMA process:
        A(q)y(t)=C(q)e(t), where
        A(q) and C(q) are defined as polinomials on q (instead of q^{-1})
        e(t) is a white noise sequence with variance: E[e(t)e(t)] = var.
    
    Parameters
    ----------
    A: numpy.ndarray
        Vector that contains the coefficients of A(q).
    C: numpy.ndarray
        Vector that contains the coefficients of C(q).
    var: float
        Variance of e(t).
    maxlag: int
        Maximum lag that will be considered on the computation of the autocorrelation (from -maxlag to +maxlag).
    
    Returns
    -------
    ryy: numpy.ndarray
        The autocorrelation function, calculated based on Soderstrom's algorithm.
    tau: numpy.ndarray
        The lag interval considered. It has the same size as ryy."""
    
    # order of A(q)
    n = A.shape[0] - 1
    # order of C(q)
    m = C.shape[0] - 1

    # calculating C(q^-1)
    cm = np.flip(C)

    # making C(q) and C(q^-1) with the same base
    zc = np.zeros((m))
    C = np.concatenate((C, zc))
    cm = np.concatenate((zc, cm))

    # calculating B(q)
    B = np.convolve(C, cm)
    # fixing the dimensions after the convolve
    B = B[m : 3 * m + 1]
    # making B(q) with the same shape as A(q)
    zb = np.zeros((n - m))
    B = np.concatenate((zb, B))
    B = np.concatenate((B, zb))
    # taking only the redundant part of B(q)
    Bn = B[0 : n + 1]
    # fliping the bn vector to find D(q)
    Bn = np.flip(Bn)
    
    # preallocating the A1 and A2 matrices
    A1 = np.zeros((n + 1, n + 1))
    A2 = np.zeros((n + 1, n + 1))

    # assembling A1 and A2
    A1 = hankel(sym.Matrix(A))
    A2 = toeplitz(sym.Matrix(A).T)
    # for k in range(0, n + 1):
        # A1[k][0 : n + 1 - k] = A[k : n + 1]
        # A2[k][k : n + 1] = A[0 : n + 1 - k]

    # assembling the matrix "calligraphic A" - to avoid redundance we'll call it Acal
    Acal = A1 + A2

    # finding the polynomial D(q)
    # D = np.linalg.solve(Acal, Bn)
    D = sym.linsolve((Acal, sym.Matrix(Bn))).args[0]
    D_ = np.array([item for item in D])
    # using the function coeff to calculate the coefficients of the correlation based on A(q) and D(q)
    ryy, tau = coeff(A, D_, maxlag)
    
    # scaling the correlation function with the variance of e(t)
    ryy = var * ryy
    
    # returning the theoretical correlation and the lag vector
    return ryy, tau

def coeff(A, D, maxlag):
     # Description to help the user
    """Function that calculates the coefficients of the theoretical correlation, based on the A(q) and D(q) polynomials,
    from -maxlag to +maxlag.
    
    Parameters
    ----------
    A: numpy.ndarray
        Vector that contains the coefficients of A(q).
    D: numpy.ndarray
        Vector that contains the coefficients of D(q).
    maxlag: int
        Maximum lag that will be considered on the computation of the autocorrelation (from -maxlag to +maxlag).
        
    Returns
    -------
    ryy: numpy.ndarray
        The autocorrelation function, calculated based on Soderstrom's algorithm.
    tau: numpy.ndarray
        The lag interval considered. It has the same size as ryy."""
    
    # preallocating the correlation function vector
    ryy = sym.zeros(maxlag + 1, 1)

    # computing the order of A(q)
    n = A.shape[0] - 1
    
    # calculating the first coefficient of the correlation function
    ryy[0] = 2 * D[0]
    # calculating the second coefficient of the correlation function
    ryy[1] = D[1] - A[1]*D[0]

    # loop that calculates the other coefficients
    for k in range(2, maxlag + 1):
        Sum = 0
        if k <= n:
            for j in range(1, k):
                Sum = Sum + A[j]*ryy[k - j]
            ryy[k] = D[k]-A[k]*D[0]-Sum
        else:
            for j in range(1, n + 1):
                Sum = Sum + A[j]*ryy[k - j]
            ryy[k] = -Sum
                
    # using the flip operation to return a vector that represents the autocorrelation from -maxlag to +maxlag
    ryyf=np.flip(ryy[1:])
    ryyc = np.concatenate((ryyf, np.array(ryy).flatten()))
        
    # calculating the size of tau
    N = 2 * maxlag + 1
    # assembling tau with linspace
    tau = np.linspace(- maxlag, maxlag, N)  # linspace(start, stop, numberofpoints)
    
    # returning
    return ryyc, tau

def arma_ccorr(A, B, C, D, var, maxlag):
    # Description to help the user
    """Function that calculates the theoretical cross correlation function of two ARMA processess:
        A(q)y(t)=B(q)e(t), and
        C(q)w(t)=D(q)e(t), where
        A(q), B(q), C(q) and D(q) are defined as polinomials on q (instead of q^{-1})
        e(t) is a white noise sequence, commom with both processes with variance: E[e(t)e(t)] = var.
        
    Parameters
    ----------
    B: numpy.ndarray
        Vector that contains the coefficients of B(q) on a "q basis".
    A: numpy.ndarray
        Vector that contains the coefficients of A(q) on a "q basis".
    D: numpy.ndarray
        Vector that contains the coefficients of D(q) on a "q basis".
    C: numpy.ndarray
        Vector that contains the coefficients of C(q) on a "q basis".
    var: float
        Variance of e(t).
    maxlag: int
        Maximum lag that will be considered on the computation of the cross correlation (from -maxlag to +maxlag).
            
    Returns
    -------
    ryw: numpy.ndarray
        The cross correlation function, calculated based on Soderstrom's algorithm.
    tau: numpy.ndarray
        The lag interval considered. It has the same size as ryw."""
        
    # transforms A(q) and B(q) to the same polynomial base
    # order of A(q)
    n = A.shape[0] - 1
    # order of B(q)
    nb = B.shape[0] - 1
    # add zeros to B(q)
    ndif = n - nb
    zb = np.zeros((ndif))
    B = np.concatenate((zb, B))

    # transforms C(q) and D(q) to the same polynomial base
    # order of C(q)
    m = C.shape[0] - 1
    # order of D(q)
    md = D.shape[0] - 1
    # add zeros to D(q)
    mdif = m - md
    zd = np.zeros((mdif))
    D = np.concatenate((zd, D))
    
    # creates the lag vector
    N = 2 * maxlag + 1
    tau = np.linspace(- maxlag, maxlag, N)  # linspace(start, stop, numberofpoints)
    
    # transforms B(q) and D(q) to the same polynomial base to use np.convolve()
    B = np.concatenate((B, np.zeros((m))))
    D = np.flip(D)
    D = np.concatenate((np.zeros((n)), D))
    # polynomial multiplication with np.convolve()
    H = np.convolve(B, D)
    # fixing the dimension after np.convolve()
    H = H[n : 2 * n + m + 1]

    # assembling the linear equations
    M1 = sym.zeros(n + m + 1, n + m + 1)
    M2 = sym.zeros(n + m + 1, n + m + 1)

    # loop that assembles the equations related to the F(q) unknown
    for k in range (0, n + 1):
        M1[k : k + m + 1, k] = np.flip(C)
    
    # loop that assembles the equations related to the G(q^-1) unknown
    for j in range (0, m):
        M2[j : j + n + 1, n + j + 1] = A.T
        
    # sum M1 and M2 to produce the full matrix of the system    
    M = M1 + M2

    # solving the linear system of equations
    # x = np.linalg.solve(M, H)
    x = np.array(sym.linsolve((M, sym.Matrix(H))).args[0]).flatten()

    # separating the unknowns
    f = x[0 : n + 1]
    g = x[n + 1: n + m + 1]
    g = np.flip(g)
    
    # preallocating the positive and negative portion of ryw
    rywp = sym.zeros(maxlag + 1, 1)
    rywn = sym.zeros(maxlag, 1)
    
    # calculating the first coefficient of the correlation function for positive tau
    rywp[0] = f[0]
    # calculating the second coefficient of the correlation function for positive tau
    if f.shape[0] > 1:
        rywp[1] = f[1] - A[1]*rywp[0]
    
    # loop that calculates the other coefficients for positive tau
    for k in range(2, maxlag + 1):
        Sum = 0
        if k <= n:
            for j in range(1, k + 1):
                Sum = Sum + A[j] * rywp[k - j]
            rywp[k] = f[k] - Sum
        else:
            for j in range(1, n + 1):
                Sum = Sum + A[j] * rywp[k - j]
                rywp[k] = - Sum
                
    # calculates the coefficients for negative tau
    if g.shape[0] > 0:
        rywn[0] = g[0]

    # loop that calculates the other coefficients for negative tau
    for k in range(1, maxlag):
        Sum2 = 0
        if k <= m - 1:
            for j in range(1, k + 1):
                Sum2 = Sum2 + C[j] * rywn[k - j]
            rywn[k] = g[k] - Sum2
        else:
            for j in range(1, m + 1):
                Sum2 = Sum2 + C[j] * rywn[k - j]
            rywn[k] = - Sum2

    # flip the negative portion of the cross correlation function
    rywn = np.flip(rywn)
    
    # concatenate the negative and positive portions
    ryw = np.concatenate((rywn, rywp))

    # scales with the variance of e(t)
    ryw = var * ryw
    
    # returns the cross correlation function and the tau (lag) vector
    return ryw, tau