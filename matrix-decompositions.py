
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('ggplot')

# LU Decomposition and Gaussian Elimination
import numpy as np
import scipy.linalg as la
np.set_printoptions(suppress=True)

A = np.array([[1,3,4],[2,1,3],[4,1,2]])

L = np.array([[1,0,0],[2,1,0],[4,11/5,1]])
U = np.array([[1,3,4],[0,-5,-5],[0,0,-3]])
print(L.dot(U))
print(L)
print(U)

np.set_printoptions(suppress=True)

A = np.array([[1,3,4],[2,1,3],[4,1,2]])

print(A)

P, L, U = la.lu(A)
print(np.dot(P.T, A))
print
print(np.dot(L, U))
print(P)
print(L)
print(U)

# Cholesky Decomposition

A = np.array([[1,3,5],[3,13,23],[5,23,42]])
L = la.cholesky(A)
print(np.dot(L.T, L))

print(L)
print(A)

# Matrix Decompositions for PCA and Least Squares
# Eigendecomposition

A = np.array([[0,1,1],[2,1,0],[3,4,5]])

u, V = la.eig(A)
print(np.dot(V,np.dot(np.diag(u), la.inv(V))))
print(u)


A = np.array([[0,1],[-1,0]])
print(A)

u, V = la.eig(A)
print(np.dot(V,np.dot(np.diag(u), la.inv(V))))
print(u)

A = np.array([[0,1,1],[2,1,0],[3,4,5]])
u, V = la.eig(A)
print(u)
print np.real_if_close(u)


A = np.array([[8,6,4,1],[1,4,5,1],[8,4,1,1],[1,4,3,6]])
b = np.array([19,11,14,14])
la.solve(A,b)
b = np.array([19.01,11.05,14.07,14.05])
la.solve(A,b)


U, s, V = np.linalg.svd(A)
print(s)
print(max(s)/min(s))


# QR 
def qr(A):
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A
 
def make_householder(a):
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H
 
# task 1: show qr decomp of wp example
a = np.array(((
    (12, -51,   4),
    ( 6, 167, -68),
    (-4,  24, -41),
)))
 
q, r = qr(a)
print('q:\n', q.round(6))
print('r:\n', r.round(6))
 
# task 2: use qr decomp for polynomial regression example
def polyfit(x, y, n):
    return lsqr(x[:, None]**np.arange(n + 1), y.T)
 
def lsqr(a, b):
    q, r = qr(a)
    _, n = r.shape
    return np.linalg.solve(r[:n, :], np.dot(q.T, b)[:n])
 
x = np.array((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
y = np.array((1, 6, 17, 34, 57, 86, 121, 162, 209, 262, 321))
 
print('\npolyfit:\n', polyfit(x, y, 2))
