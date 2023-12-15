from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as ham



figure, ax = plt.subplots(figsize=(10, 10))

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
        somme+=np.where(y[i]*np.dot(w, X[i, :])<=0,1,0)
    return somme/n  

#POUR L'ANIMATION DES TROIS SEPARATEURS____________________________________________________________________________________________
lines0,lines1,lines2,lines3,lines4,lines5,lines6,lines7,lines8,lines9,lines10,lines11,lines12,lines13,lines14=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]    
liness=[lines0,lines1,lines2,lines3,lines4,lines5,lines6,lines7,lines8,lines9,lines10,lines11,lines12,lines13,lines14]

#DATA AVEC TROIS CENTRES DIFFERENTS________________________________________________________________________________________________
centers=[[-7, 0], [1, 15], [3, -3],[9,3],[-10,8]]
X, Y = datasets.make_blobs(n_samples=102,n_features=2, centers=centers,cluster_std=0.8, random_state=8)

np.random.seed(5)



#LE PERCEPTRON Pocket Pour ECOC______________________________________________________________________________________________________________________
def PLA(X, y,j,w):
        n_samples = X.shape[0]
        if j>=n_samples:
            j=0
        for k in range(j,n_samples):
            if y[k]*np.dot(w, X[k, :]) <= 0:
                w += y[k]*X[k, :]
                print(w , "PLA") 
                break
            break         
        return w,k+1     
        

def perceptronPocket(X, y, n_iter,lines):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        weights = np.zeros((n_features+1,))
        X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        x = np.linspace(-10,10,40)
       
        Ws=weights
        Wt=weights
        lines=ax.plot(x, ff(x,Ws))
        j=0
        for i in range(1,n_iter):
            Wt,j=PLA(X,y,j,Wt)
            #print(Wt, "Wt")
            Loss=loss(n_samples,Wt,X,y)
            if Loss<=loss(n_samples,Ws,X,y):
                lines=[]
                plt.pause(0.000000000001)
                ax.lines.pop(0)
                lines=ax.plot(x, ff(x,Wt))
                Ws=Wt
        return Ws,lines
                    

#POUR LE PLOT DES DIFERENTES CLASSES________________________________________________________________________________________________

def plot_data_points(ax, X, y):
    neg_class = (y == -1)
    pos_class = (y == 1)
    ax.scatter(X[neg_class, 0] ,X[neg_class, 1])
    ax.scatter(X[pos_class, 0] ,X[pos_class, 1])
    plt.axis('scaled')

#LA FONCTION DE ECOC____________________________________________________________________________________________________________

W=[]



def ECOC_MATRICE_EXHAUSTIVE(k):
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
    table[table.shape[0]-1][14]=0
    return table    
            
matrice=ECOC_MATRICE_EXHAUSTIVE(5)




def EXAUSTIVE_ECOC(matrice,X,Y):
    for i in range(matrice.shape[1]):
        y=(matrice.T)[i].copy()
        for j in range(len(y)):
            if y[j]==0:
                y[j]=j
            else:
                y[j]=10
                
                  
        labels=Y.copy()
        for j in range(0,102):
            if labels[j] in y:
                labels[j]=-1
            else:
                labels[j]=1


        pos_class = (labels == 1)

        neg_class = (labels == -1 )




        ax.scatter(X[neg_class, 0] ,X[neg_class, 1],color='blue')
        ax.scatter(X[pos_class, 0] ,X[pos_class, 1],color='red')
        plt.axis('scaled')
        x = np.linspace(-10,10,30)
        plot_data_points(ax, X, labels)
        lines=liness[i]
        w,lines=perceptronPocket(X, labels,260,lines)
        W.append(w)
        lines=[]
        plt.pause(4)
        ax.lines.pop(0)
        labels=Y.copy()
        lines=ax.plot(x, ff(x,w))
        ax.cla()
    return W

        
        
        
def DECODING(x,w,matrice):
    code=list()
    for i in range(matrice.shape[1]):
        if sign(W[i].dot(x))==-1:
            code.append(0)
        else:
            code.append(1)
    code=np.array(code)
    dmin=10000
    for i in range(matrice.shape[0]):
        d=ham.hamming(code,matrice[i])
        if dmin>d:
            dmin=d
            classe=i
    return dmin,classe        
             
        
        


        
        
    
W=EXAUSTIVE_ECOC(matrice,X,Y)


class1 = ( Y== 0)
class2 = (Y == 1)
class3 = (Y== 2)
class4 = (Y== 3)
class5 = (Y== 4)



for i in range(-10,10,1):
    ax.scatter(X[class1, 0] ,X[class1, 1],color='blue',label ='classe 1')
    ax.scatter(X[class2, 0] ,X[class2, 1],color='red',label ='classe 2')  
    ax.scatter(X[class3, 0] ,X[class3, 1],color='green',label ='classe 3')
    ax.scatter(X[class4, 0] ,X[class4, 1],color='brown',label ='classe 4')
    ax.scatter(X[class5, 0] ,X[class5, 1],color='yellow',label ='classe 5')
    j=np.random.randint(0, 15)
    x=np.array([i,j,1])
    distance,classe=DECODING(x,W,matrice)
    ax.set_title("Le point x appartient a classe %s" %(classe+1))
    ax.scatter(x[0],x[1],color='black',label ='point black')
    plt.legend()
    plt.pause(3.5)
    ax.cla()

plt.axis('scaled')
plt.show()
