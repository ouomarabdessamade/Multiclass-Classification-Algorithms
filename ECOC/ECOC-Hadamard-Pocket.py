from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import math
from pylab import *


figure, ax = plt.subplots()

#POUR L'ANIMATION DES TROIS SEPARATEURS____________________________________________________________________________________________
lines0,lines1,lines2,lines3=[],[],[],[]   
liness=[lines0,lines1,lines2,lines3]

#DATA AVEC TROIS CENTRES DIFFERENTS________________________________________________________________________________________________
centers=[[-7, 0], [1, 15], [6, -3], [0,3]]
X, Y = datasets.make_blobs(n_samples=300, n_features=2, centers=centers,cluster_std=0.8, random_state=8)

np.random.seed(5)


#EQUATION DU SEPARATEUR PAR PERCEPTRON____________________________________________________________________________________________
def ff(x,w):
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
        return w,k+1     
        

def perceptronPocket(X, y, n_iter,lines):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        weights = np.zeros((n_features+1,))
        X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        x = np.linspace(-10, 15, 2)
       
        Ws = weights
        Wt = weights
        lines=ax.plot(x, ff(x,Ws))
        j=0
        for i in range(1, n_iter):
            Wt, j = PLA(X, y, j, Wt)
            #print(Wt, "Wt")
            Loss = loss(n_samples,Wt,X,y)
            if Loss<=loss(n_samples, Ws, X, y):
                lines=[]
                plt.pause(0.05)
                ax.lines.pop(0)
                lines=ax.plot(x, ff(x,Wt))
                Ws=Wt
        return Ws,lines
    
    
def EXAUSTIVE_ECOC(matrice, X, Y):
    m = len(X)
    labels = Y.copy()
    x = np.linspace(-10, 15, 2)
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
        plt.grid(False)#plot a grid
        plt.xlim(-10, 15)
        plt.ylim(-10, 20)
        ax.scatter(X[neg_class, 0] ,X[neg_class, 1], color='blue', label='class 1')
        ax.scatter(X[pos_class, 0] ,X[pos_class, 1], color='green', label='class 2') 
        
        lines = liness[i]
        w, lines = perceptronPocket(X, labels, 350, lines)
        W.append(w)
        lines=[]
        ax.lines.pop(0)
        labels= Y.copy()
        lines = ax.plot(x, ff(x, w))
        plt.legend()
        plt.pause(0.05)

        ax.cla()
    
    
    return W
  
             

theta = EXAUSTIVE_ECOC(MatricHadamard, X, Y)

w1 = theta[0]
w2 = theta[1]
w3 = theta[2]
w4 = theta[3]
x = np.linspace(-10, 15, 2)
plt.title('les resultats finale de ECOC-Hadamard')
plt.grid(False)#plot a grid
plt.xlim(-10, 15)
plt.ylim(-10, 20)

ax.scatter(X[:,: -1] ,X[:,-1], c =Y)

ax.plot(x, ff(x,w1), label='Separateur 1')
ax.plot(x, ff(x,w2), label='Separateur 2')
ax.plot(x, ff(x,w3), label='Separateur 3')
ax.plot(x, ff(x,w4), label='Separateur 4')



plt.legend()
plt.show()

#import scipy.spatial.distance as ham
# def DECODING(x,w,matrice):
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
