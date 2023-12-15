from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import math
from pylab import *





figure, ax = plt.subplots()

#POUR L'ANIMATION DES TROIS SEPARATEURS____________________________________________________________________________________________
lines0,lines1,lines2,lines3=[],[],[],[]   
liness=[lines0,lines1, lines2, lines3]

#DATA AVEC TROIS CENTRES DIFFERENTS________________________________________________________________________________________________
centers=[[-7, 0], [1, 15], [6, -3], [0,5]]
X, Y = datasets.make_blobs(n_samples=102, n_features=2, centers=centers,cluster_std=0.8, random_state=8)

np.random.seed(5)

#EQUATION DU SEPARATEUR PAR PERCEPTRON____________________________________________________________________________________________
def ff(x, w):
    if w[1]==0:
        w5=0.001
        return (-w[2]-w[0]*x)/w5
    return (-w[2]-w[0]*x)/w[1]

#LOSS FUNCTION____________________________________________________________________________________________________________________
def loss(n,w,X,y):
    somme=0
    for i in range(0,n):
        somme+= np.where(y[i]*np.dot(w, X[i, :])<=0,1,0)
    return somme/n  

#-----------------une fonction pour ploter les resultats final----------------#
def PlotSeparateurFinal(X, y, W):
    n_x = 0.15
    n_y = 0.15
    x1 = np.arange(-10.0, 15.0, n_x)
    x2 = np.arange(-10.0, 15.0, n_y)
    xx1, xx2 = np.meshgrid(x1, x2)
    
    f1 = W[0]
    f2 = W[1]
    f3 = W[2]
    f4 = W[3]
    
    plt.grid(False)#plot a grid
    plt.xlim(-15, 20)
    plt.ylim(-15, 25)
    
    cs1 = ax.contour(x1, x2, f1, 0, linewidths=2, colors='k')
    ax.clabel(cs1, inline=1, fontsize=20, fmt='%1.1f', manual=[(1,10)], use_clabeltext='classe 1') 
    
    cs2 = ax.contour(x1, x2, f2, 0, linewidths=2,colors='r')
    ax.clabel(cs2, inline=1, fontsize=20, fmt='%1.1f', manual=[(1,1)], use_clabeltext='classe 2') 
    
    cs3 = ax.contour(x1, x2, f3, 0, linewidths=2, colors='g')
    ax.clabel(cs3, inline=1, fontsize=20, fmt='%1.1f', manual=[(1,1)], use_clabeltext='classe 3') 
    
    cs4 = ax.contour(x1, x2, f4, 0, linewidths=2, colors='b')
    ax.clabel(cs4, inline=1, fontsize=20, fmt='%1.1f', manual=[(1,1)], use_clabeltext='classe 4') 
    
    ax.scatter(X[:,: -1] ,X[:,-1], c =y)
    
    plt.show()
    

'''   
#---la visualisation de data-------------#
ax.scatter(X[:,: -1] ,X[:,-1], c =Y)
'''

#---------la fonction hadamard : pour construire la matrice de hadamard----------------------#
def hadamardMatrix(n):
    if n < 1:
        lg2 = 0
    else:
        lg2 = int(math.log(n, 2))
    if 2 ** lg2 != n:
        print('le dim doit eter paire SVP !')
    else :
        H = np.array([[1]])

        for i in range(0, lg2):
            H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))

    return H
    
#print(hadamardMatrix(4))

MatricHadamard = hadamardMatrix(4)

W=[]
#LE PERCEPTRON Pocket Pour ECOC______________________________________________________________________________________________________________________
def PLA(X, y, j, w):
        n_samples = X.shape[0]
        if j>=n_samples:
            j=0
        for k in range(j,n_samples):
            if y[k]*np.dot(w, X[k, :]) <= 0:
                w += y[k]*X[k, :]
                #print(w , "PLA") 
                break
            break         
        return w, k+1     
        
#LE KERNEL______________________________________________________________________________________________________________________
def polynomial_kernel1(X, x, p=2):
    somme=list()
    for i in range(X.shape[0]):
        somme.append((1 + np.dot(X[i], x[i])) ** p)
    somme=np.array(somme)
    return somme

def polynomial_kernel(x, y, p=2):
    return (1 + np.dot(x, y)) ** p

def kernelsum(alpha,y,i,K_P,n):
    somme=0
    for j in range(n):
        somme=somme+alpha[j]*y[j]*K_P[i,j]
    return somme  


def kernelDecision(X,x,alpha,n,y):
    somme=0
    for i in range(n):
        somme=somme+alpha[i]*y[i]*polynomial_kernel(X[i], x)
    if somme<0:
        classe=-1
    else:
        classe=1
    return classe  


    
    
#LE PERCEPTRON AVEC KERNEL______________________________________________________________________________________________________________________
def PERCEPTRON_KERNEL(X,y,n):
    K_P = np.zeros((n, n))
    alpha = np.zeros((n,))
    for i in range(n):
        for j in range(n):
            K_P[i,j] = polynomial_kernel(X[i], X[j],2)
    m = 40
    b = 0
    j = 0        
    while(j<=m):
        for i in range(n):
            if y[i]*(kernelsum(alpha,y,i,K_P,n)+b)<=0:
                alpha[i]=alpha[i]+1
                b=b+y[i]   
        j=j+1  
        
    n_x = 0.15
    n_y = 0.15
    x1 = np.arange(-10.0, 15.0, n_x)
    x2 = np.arange(-10.0, 15.0, n_y)
    xx1, xx2 = np.meshgrid(x1, x2)

    f = np.zeros(xx1.shape)
    for i in range(xx1.shape[0]):
        for j in range(xx1.shape[1]):
            
            f[i,j] = (
                        (
                            alpha              
                            *y
                            * polynomial_kernel1(X, np.tile(np.array([xx1[i,j],xx2[i,j]]),(X.shape[0],1)),2) 
                        ).sum()
                     ) 
    
    cs = ax.contour(x1, x2, f, 0, linewidths=2,colors='k')
    ax.clabel(cs, inline=1, fontsize=20, fmt='%1.1f', manual=[(1,1)]) 
    plt.pause(0.05)
    return f

#----------------------ECOC Algorithme----------------------------------------#
def ECOC_Algorithme(matrice, X, Y):
    listee = []
    labels = Y.copy()
    Les_Y = list()
    m = len(X)
    
    for i in range(matrice.shape[1]):
        y = (matrice.T)[i].copy()
        #print(y)
        for j in range(len(y)):
            if y[j] == 1 :
                y[j] = j
            else:
                y[j] = 10
                
        for j in range(0, m):
            if labels[j] in y:
                labels[j] = -1
            else:
                labels[j] = 1
        
        pos_class = (labels == 1)
        neg_class = (labels == -1)
        
        Les_labels=labels.copy()
        Les_Y.append(Les_labels)

        ax.scatter(X[neg_class, 0] ,X[neg_class, 1], color='blue')
        ax.scatter(X[pos_class, 0] ,X[pos_class, 1], color='green')
        
        fonc = PERCEPTRON_KERNEL(X, labels, 102)

        listee.append(fonc)
        
        labels=Y.copy()
        ax.cla()
    
    return PlotSeparateurFinal(X, Y, listee)

ECOC_Algorithme(MatricHadamard, X, Y)

#import scipy.spatial.distance as ham
# def DECODING(x, w, matrice):
#     code=list()
#     for i in range(matrice.shape[1]):
#         if sign(W[i].dot(x))==-1:
#             code.append(0)
#         else:
#             code.append(1)
#     code=np.array(code)
#     dmin=10000
#     for i in range(matrice.shape[0]):
#         d=ham.hamming(code,matrice[i])
#         if dmin>d:
#             dmin=d
#             classe=i
#     return dmin,classe        
             





