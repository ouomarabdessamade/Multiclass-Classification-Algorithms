from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np



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
centers=[[-7, 0], [1, 7], [4, -1]]
X, Y = datasets.make_blobs(n_samples=102,n_features=2, centers=centers,cluster_std=0.8, random_state=8)

np.random.seed(5)


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

#LA FONCTION DE ONE VS ALL____________________________________________________________________________________________________________

W=[]
def OnevsAll(classesNum,X,Y):
    labels=Y.copy()
    for i in range(classesNum):
        
        print("before",labels)
        for j in range(0,102):
            if labels[j]==i :
                labels[j]=1
            else:
                labels[j]=-1
        
        print ( "Iteration :", i ,"here it is ", labels )
        pos_class = (labels == 1)

        neg_class = (labels == -1 )

        

        
        ax.scatter(X[neg_class, 0] ,X[neg_class, 1],color='blue')
        ax.scatter(X[pos_class, 0] ,X[pos_class, 1],color='red')
        plt.axis('scaled')
        x = np.linspace(-10,10,30)
        plot_data_points(ax, X, labels)
        lines=liness[i]
        w,lines=perceptron(X,labels,lines)
        W.append(w)
        lines=[]
        plt.pause(3)
        ax.lines.pop(0)
        labels=Y.copy()
        lines=ax.plot(x, f(x,w))
        ax.cla()
    return W
        
        
     
        


W=OnevsAll(3,X,Y) 
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
plt.show()  


#POUR LE CAS DE POCKET

'''

def PLA(X, y,w):
        n = X.shape[0]
        for i in range(0,n):
            if y[i]*np.dot(w, X[i, :])<=0:
                w=w+y[i]*X[i, :]
                print(w ,"PLA")
                break
            break
        return w

def PLA(X, y,j,w):
        n_samples = X.shape[0]
        for i in range(j,n_samples):
            if y[j]*np.dot(w, X[j, :]) <= 0:
                w += y[j]*X[j, :]
                print(w , "PLA") 
                break
            break         
        return w,i+1     
        

def perceptronPocket(X, y, n_iter,lines):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        weights = np.zeros((n_features+1,))
        X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        x = np.linspace(-3,7,20)
       
        Ws,Wt=weights,weights
        x = np.linspace(-3,4,40)
        lines=ax.plot(x, f(x,Ws))
        j=0
        for i in range(1,n_iter):
            Wt,j=PLA(X,y,j,Wt)
            print(Wt, "Wt")
            Loss=loss(n_samples,Wt,X,y)
            if Loss<=loss(n_samples,Ws,X,y):
                lines=[]
                plt.pause(0.001)
                ax.lines.pop(0)
                lines=ax.plot(x, f(x,Wt))
                Ws=Wt
        return Ws
                    
             


'''



