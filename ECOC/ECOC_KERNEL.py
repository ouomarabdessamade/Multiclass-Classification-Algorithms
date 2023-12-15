
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as ham



plt.style.use('seaborn-whitegrid')
plt.rcParams['contour.negative_linestyle'] = 'solid'

import pandas as pd
data=pd.read_csv(r"train.csv")
Y=np.array(data["Label"])
X=np.array(data[["X","Y"]])

figure, ax = plt.subplots(figsize=(10, 10))


#POUR L'ANIMATION DES TROIS SEPARATEURS____________________________________________________________________________________________
lines0,lines1,lines2,lines3,lines4,lines5,lines6,lines7,lines8,lines9,lines10,lines11,lines12,lines13,lines14=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]    
liness=[lines0,lines1,lines2,lines3,lines4,lines5,lines6,lines7,lines8,lines9,lines10,lines11,lines12,lines13,lines14]

'''
#DATA AVEC TROIS CENTRES DIFFERENTS________________________________________________________________________________________________
centers=[[-7, 0], [1, 15], [3, -3],[9,3]] #[[-7, 0], [1, 15], [3, -3],[9,3],[-10,8]]
X, Y = datasets.make_blobs(n_samples=102,n_features=2, centers=centers,cluster_std=0.8, random_state=8)

np.random.seed(5)
'''

#LE KERNEL______________________________________________________________________________________________________________________
def polynomial_kernel1(X, x, p):
    somme=list()
    for i in range(X.shape[0]):
        somme.append((1 + np.dot(X[i], x[i])) ** p)
    somme=np.array(somme)
    return somme

def polynomial_kernel(x, y, p):
    return (1 + np.dot(x, y)) ** p

def kernelsum(alpha,y,i,K_P,n):
    somme=0
    for j in range(n):
        somme=somme+alpha[j]*y[j]*K_P[i,j]
    return somme  


def kernelDecision(X,x,alpha,n,y):
    somme=0
    for i in range(n):
        somme=somme+alpha[i]*y[i]*polynomial_kernel(X[i], x,9)
    if somme<0:
        classe=-1
    else:
        classe=1
    return classe  



#LE PERCEPTRON AVEC KERNEL______________________________________________________________________________________________________________________
def PERCEPTRON_KERNEL(X,y,n):
    K_P = np.zeros((n, n))
    alpha=np.zeros((n,))
    for i in range(n):
        for j in range(n):
            K_P[i,j] = polynomial_kernel(X[i], X[j],9)
    m=110
    b=0
    j=0        
    while(j<=m):
        for i in range(n):
            if y[i]*(kernelsum(alpha,y,i,K_P,n)+b)<=0:
                alpha[i]=alpha[i]+1
                b=b+y[i]   
        j=j+1  
        
    
    n_x = 0.04
    n_y = 0.04
    x1 = np.arange(-1.0, 1.0, n_x)
    x2 = np.arange(-1.0, 1.0, n_y)
    xx1, xx2 = np.meshgrid(x1, x2)

    f = np.zeros(xx1.shape)
    for i in range(xx1.shape[0]):
        for j in range(xx1.shape[1]):
            
            f[i,j] = (
                        (
                            alpha              
                            *y
                            *polynomial_kernel1(X, np.tile(np.array([xx1[i,j],xx2[i,j]]),(X.shape[0],1)),9) 
                        ).sum()
                     ) 
    
    cs = ax.contour(x1, x2, f, 0, linewidths=2,colors='k')
    #ax.clabel(cs, inline=1, fontsize=15, fmt='%1.1f', manual=[(1,1)]) 
    ax.set_title("EXHAUSTIVE MATRIX METHOD \n Perceptron Kernel Polynomial de Degre 9 ")
    plt.pause(3)
    return alpha
        


#POUR LE PLOT DES DIFERENTES CLASSES________________________________________________________________________________________________

def plot_data_points(ax, X, y):
    neg_class = (y == -1)
    pos_class = (y == 1)
    ax.scatter(X[neg_class, 0] ,X[neg_class, 1],label ='Classe Negative')
    ax.scatter(X[pos_class, 0] ,X[pos_class, 1],label ='Classe Positive')
    plt.legend()
    plt.axis('scaled')

#LES FONCTIONS DE ECOC____________________________________________________________________________________________________________

W=[]
#np.random.seed(5)
#ORTHOGONAL ARRAYS OF CASES 4 ET 8__________________________________________________________________________________________________
def ECOC_OA(N):
    if N==4:
        X=np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
    elif N==8:
        X=np.array([[0,0,0,0,0],[1,0,0,1,1],[0,1,0,1,0],[0,0,1,0,1],
                    [1,1,0,0,1],[1,0,1,1,0],[0,1,1,1,1],[1,1,1,0,0]])
    return X             
#_____________________________________________________________________________________________________________________________________

def ECOC_EXAUSTIVE(k):
    table=np.ones((np.power(2,k-1)-1,))
    cst=k
    while cst>1:
        a=np.ones((np.power(2,k-1)-1,))
        table=np.vstack((table, a))
        cst=cst-1
    cst2=1
    n=2
    while cst2<k:
        for i in range(0,np.power(2,k-1)-1,2*np.power(2,k-n)):
            if(np.power(2,k-n)+i!=15):
                for j in range(i,i+np.power(2,k-n),1):
                    table[cst2][j]=0
                
        cst2=cst2+1
        n=n+1
    #table[table.shape[0]-1][14]=0 #Use it for 5 classes and more
    return table   
 
            
#matrice=ECOC_OA(4)
matrice=ECOC_EXAUSTIVE(3)


def DECODING(x,matrice,n,alpha,y):
    code=list()
    
    for i in range(matrice.shape[1]):
        if kernelDecision(X,x,alpha[i],n,y[i])==-1:
            code.append(0)
        else:
            code.append(1)
    print(code)
    code=np.array(code)
    dmin=10000
    for i in range(matrice.shape[0]):
        d=ham.hamming(code,matrice[i])
        print(matrice[i])
        if dmin>d:
            dmin=d
            classe=i
    return dmin,classe        
             
        
        

def EXAUSTIVE_ECOC(matrice,X,Y,n):
    Les_Y,alphak=list(),list()
    for i in range(matrice.shape[1]):
        y=(matrice.T)[i].copy()
        for j in range(len(y)):
            if y[j]==0:
                y[j]=j
            else:
                y[j]=10
                
                  
        labels=Y.copy()
        for j in range(0,n):
            if labels[j] in y:
                labels[j]=-1
            else:
                labels[j]=1

        
        pos_class = (labels == 1)

        neg_class = (labels == -1 )



        Les_labels=labels.copy()
        Les_Y.append(Les_labels)

        ax.scatter(X[neg_class, 0] ,X[neg_class, 1],color='blue')
        ax.scatter(X[pos_class, 0] ,X[pos_class, 1],color='red')
        plt.axis('scaled')
        x = np.linspace(-10,10,30)
        plot_data_points(ax, X, labels)
        #lines=liness[i]
        alpha=PERCEPTRON_KERNEL(X,labels,n)

        alphak.append(alpha)
        labels=Y.copy()
        ax.cla()
    
    return alphak,Les_Y

        
        
        
    
alpha,labels=EXAUSTIVE_ECOC(matrice,X,Y,300)
print("labels :" , labels)
class1 = ( Y== 0)
class2 = (Y == 1)
class3 = (Y== 2)
#class4 = (Y== 3)
#class5 = (Y== 4)

colors=['blue','red','green']
m=0
for i in np.arange(-1,1.5,0.33):
    ax.scatter(X[class1, 0] ,X[class1, 1],color='blue',label ='classe 1')
    ax.scatter(X[class2, 0] ,X[class2, 1],color='red',label ='classe 2')  
    ax.scatter(X[class3, 0] ,X[class3, 1],color='green',label ='classe 3')
    #ax.scatter(X[class4, 0] ,X[class4, 1],color='brown',label ='classe 4')
    #ax.scatter(X[class5, 0] ,X[class5, 1],color='yellow',label ='classe 5')
    j=np.random.randint(-5, 5)/5
    x=np.array([i,j])
    print("pour x = " , x)
    distance,classe=DECODING(x,matrice,300,alpha,labels)
    ax.set_title("LA PHASE DE TESTING \n Le point x appartient a la classe %s" %(classe+1),fontsize=15,c=colors[classe])
    ax.scatter(x[0],x[1],color='black',marker='x',s=120.0,label ='Crois Noir')
    plt.legend()
    plt.pause(4.5)
    ax.cla()

plt.axis('scaled')
plt.show()


