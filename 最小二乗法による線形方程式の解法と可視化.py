# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

A = np.array([[1, 1],
              [2, -1],
              [-2, 4]])
b = np.array([1, 2, 7])

# numpy.linalg.lstsq を使って解を求める
x0, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)

print("最小二乗法による解 x =", x0)
print("残差の二乗和 residuals =", residuals)
print("Aのランク rank =", rank)
print("Aの特異値 singular_values =", singular_values)

#正規方程式を構成し、 逆行列を使って解を求める
AtA = np.dot(A.T, A)    
Atb = np.dot(A.T, b)    

x1 = np.dot(np.linalg.inv(AtA), Atb)

print("正規方程式による課題２の解 x =", x1)

#Aを特異値分解して擬似逆行列を求め、解を求める

U, s, VT = np.linalg.svd(A)

S_inv = np.zeros((2, 3))
for i in range(len(s)):
    S_inv[i, i] = 1 / s[i]

A_pinv = np.dot(VT.T, np.dot(S_inv, U.T))

x2 = np.dot(A_pinv, b)

print("特異値分解を使った課題３の解 x =", x2)

#ベクトル{b,p,a1,a2}の関係を図示
p = np.dot(A, x0)

a1 = A[:, 0]
a2 = A[:, 1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

origin = np.zeros(3)

ax.quiver(*origin, *b, color='r', label='b', linewidth=2)
ax.quiver(*origin, *p, color='g', label='p (projection)', linewidth=2)
ax.quiver(*origin, *a1, color='b', label='a1', linewidth=2)
ax.quiver(*origin, *a2, color='m', label='a2', linewidth=2)

ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 10])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.title('ベクトル b, p, a1, a2 の関係')
plt.show()
