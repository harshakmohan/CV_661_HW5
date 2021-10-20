import numpy as np
import enum

K = np.array([[100, 0, 320],[0, 100, 240],[0, 0, 1]])
Rt = np.array([[-1,0,0,1],[0,-1,0,1],[0,0,1,-1]])

P = np.matmul(K,Rt)
print('P:\n', P)

X = [[0,0,5],[1,0,7],[1,1,8],[-6,8,8],[2,4,10],[-3,8,8]]
print(np.asarray(X))

x = []
for x_i in X:
    x_i.append(1)
    x_i = np.asarray(x_i)
    x_i = np.transpose(x_i)
    print(x_i.shape)
    x.append(np.matmul(P,x_i))

x = np.asarray(x)

print('computed x:\n', x)

A = []
for i, X_i in enumerate(X):

    j = np.concatenate((np.dot(-x[i][2],X_i), np.dot(x[i][1],X_i)))
    A.append(np.concatenate((np.zeros(4), j)))

    k = np.concatenate((np.zeros(4), np.dot(-x[i][0],X_i)))
    A.append(np.concatenate((np.dot(x[i][2],X_i) , k)))

A = np.asarray(A)
print('A:', A.shape)

U, S, VH = np.linalg.svd(A)

print('V: \n', VH)
