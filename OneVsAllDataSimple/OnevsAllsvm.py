from scipy.optimize import minimize
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from pylab import *




figure, ax = plt.subplots(figsize=(10, 10))

#EQUATION DU SEPARATEUR PAR PERCEPTRON____________________________________________________________________________________________
def f(x,w):
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
lines0,lines1,lines2=[],[],[]      
liness=[lines0,lines1,lines2]

#DATA AVEC TROIS CENTRES DIFFERENTS________________________________________________________________________________________________
centers=[[-1, 0], [1, 7], [8, -1]]
X, Y = datasets.make_blobs(n_samples=100,n_features=2, centers=centers,cluster_std=0.8, random_state=4)

np.random.seed(21)


#LE PERCEPTRON______________________________________________________________________________________________________________________
def perceptron(X,y,lines):
    n_samples = X.shape[0]
    n=n_samples
    n_features = X.shape[1]     

    w = np.zeros((n_features+1,))

    X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
    x = np.linspace(-10,10,40)



    losss=loss(n,w,X,y)

    lines=ax.plot(x, f(x,w))
    i,t=0,0
    while losss!=0:
        losss=loss(n,w,X,y)
        for i in range(0,n):
            if y[i]*np.dot(w, X[i, :])<=0:
                lines=[]
                plt.pause(0.2)
                ax.lines.pop(0)
                print("w : " ,w,"loss : ",losss , "point mal classifie count : " ,t)
                w=w+y[i]*X[i, :]
                lines=ax.plot(x, f(x,w))
                t+=1
    return w,lines

#POUR LE PLOT DES DIFERENTES CLASSES________________________________________________________________________________________________

def plot_data_points(ax, X, y):
    neg_class = (y == -1)
    pos_class = (y == 1)
    ax.scatter(X[neg_class, 0] ,X[neg_class, 1])
    ax.scatter(X[pos_class, 0] ,X[pos_class, 1])
    plt.axis('scaled')

def SVM_HARD(X,Y):
    fun = lambda w: 0.5*(w[0]**2+w[1]**2)
    cons=list() 
    for i,j in zip(X,Y): #liste des contraintes pour le hard svm
        t = {'type': 'ineq', 'fun': lambda w,i=i,j=j:  i[0]*j*w[0] +i[1]*j*w[1]+j*w[2]-1}
        cons.append(t)
    cons = tuple(cons)

    res = minimize(fun, np.array([0, 0,0]), method='SLSQP',jac='2-point',constraints=cons) #utilisation de la fonction monimize
    w=res.x
    return w
def SVM_SOFT(x,y):
    n=100
    C=50 # on  a chosit 100 pout la constante C pour bien minimiser l'erreur
    fun = lambda w: 0.5*(w[0]**2+w[1]**2)+C*sum([ w[m] for m in range(3,n+3)])
    cons=list()
    ############################Les contraintes avec Ajout d'erreur########################################
    for i,j,k in zip(x,y,range(3,n+3)):
        f= {'type': 'ineq', 'fun': lambda w,i=i,j=j,k=k:    i[0]*j *w[0]+ i[1]*j *w[1]+ j *w[2] -1+w[k]}
        cons.append(f)
    #######################################################################################################

    #######################Les contraintes de la positivite d'erreur ########################################
    for i in range(3,n+3):
        f={'type': 'ineq', 'fun': lambda w,i=i:     w[i] }
        cons.append(f)
    #########################################################################################################
    cons = tuple(cons)

    resultat = minimize(fun, np.random.randn(1,n+3), method='SLSQP',jac='2-point',constraints=cons) #la fonction minimize

    w=resultat.x  
    return w
    

#LA FONCTION DE ONE VS ALL____________________________________________________________________________________________________________

W=[]
def OnevsAll(classesNum,X,Y,method):
    labels=Y.copy()
    for i in range(classesNum):
        
        print("before",labels)
        for j in range(0,100):
            if labels[j]==i :
                labels[j]=1
            else:
                labels[j]=-1
        
        print ( "Iteration :", i ,"here it is ", labels )
        pos_class = (labels == 1)

        neg_class = (labels == -1 )
        ax.cla()
        x = np.linspace(-10,10,40)
        ax.scatter(X[neg_class, 0] ,X[neg_class, 1],color='red')
        ax.scatter(X[pos_class, 0] ,X[pos_class, 1],color='blue')
        plt.axis('scaled')
        
        lines=liness[i]
        if method=='perceptron':
            w,lines=perceptron(X,labels,lines)
            W.append(w)
            lines=[]
            plt.pause(3)
            ax.lines.pop(0)
            labels=Y.copy()
            lines=ax.plot(x, f(x,w))
            ax.cla()
        elif(method=='hard svc'):
            w=SVM_HARD(X,labels)
            print(w,"ici")
            W.append(w)
            labels=Y.copy()
            x2 = (-W[i][1]*x-W[i][2])/W[i][0]
            x4 = (-W[i][1]*x-W[i][2]-1)/W[i][0]
            x6 = (-W[i][1]*x-W[i][2]+1)/W[i][0]
            plt.title('Hard_SVM')
            ax.plot(x, x2,c = 'black')
            ax.plot(x, x4,'-.',c = 'yellow')
            ax.plot(x, x6,'-.',c = 'yellow')   
            plt.pause(3)
            ax.cla()
        else:
            w=SVM_SOFT(X,labels).copy()
            print(w,"ici")
            W.append(w)
            labels=Y.copy()
            x2 = (-W[i][1]*x-W[i][2])/W[i][0]
            x4 = (-W[i][1]*x-W[i][2]-1)/W[i][0]
            x6 = (-W[i][1]*x-W[i][2]+1)/W[i][0]
            plt.title('SOFT_SVM')
            ax.plot(x, x2,c = 'black')
            ax.plot(x, x4,'-.',c = 'yellow')
            ax.plot(x, x6,'-.',c = 'yellow')    
            plt.pause(3)
            
            

    return W
        
        
     
        


W=OnevsAll(3,X,Y,'soft svc') 
'''
class1 = ( Y== 0)
class2 = (Y == 1)
class3 = (Y== 2)
ax.scatter(X[class1, 0] ,X[class1, 1],color='blue')
ax.scatter(X[class2, 0] ,X[class2, 1],color='red')  
ax.scatter(X[class3, 0] ,X[class3, 1],color='green')
plt.axis('scaled')
x = np.linspace(-10,10,30)      

for i in range(3):
    plt.pause(1)
    ax.plot(x, f(x,W[i]))
plt.pause(6)  
'''
ax.set_xlim([-2, 7])
ax.set_ylim([-5, 10])
plt.show()  





