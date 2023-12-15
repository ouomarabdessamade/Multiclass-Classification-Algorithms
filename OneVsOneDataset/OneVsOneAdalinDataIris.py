import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



df = pd.read_csv('IRIS.csv')

x1 = df['sepal_width']
x2 = df['petal_width']
#Concatener x1 et x2
X = pd.concat([x1, x2], axis=1)


#transformer la colonne target (Y en 1 et N en 0)
target_value = { 'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica' : 3}
target = df['species'] #la colonne species est notre variable cible
#on le supprime
df.drop('species', axis=1, inplace= True)
#on remplace chaque categories des nomber (1 , 2 et 3)
target = target.map(target_value)
#si on affiche maintenant la variable target :
#target


#--------------data
data = pd.concat([X, target], axis=1)
data = np.array(data)
#-----------la visualisation  de data
#plt.scatter(df['sepal_width'], df['petal_width'], c =target)


#-----------------------------loss function-----------------------------------#
def calculdelsw(w, x):
    som = 0
    for i in range(len(x)):
        som+= (x[i][2] - (np.dot(np.transpose(w), np.array([x[i][0], x[i][1], 1]))))**2
    return som/len(x)



#-----------------------------le gradient-------------------------------------#
def gradient(w , x):
    #print(x)
    som = 0
    for i in range(len(x)):
        ei = (x[i][2] - np.dot(np.transpose(w), np.array([x[i][0], x[i][1], 1])))
        #print("ei = ", ei)
        som += ei * (np.array([x[i][0], x[i][1], 1]))
    result = (-2*som)/len(x)
    return np.linalg.norm(result) 



#------------------------Algorithme de Adalin----------------------------------#
def AdalinOneVsOne(w, x, alpha, col, delta, visualise ):
    plt.ion()
    compteur = 0
    gradLosw = 1
    #alpha = alphaSearch(w, x)
    grad =  gradient(w , x)
    #print("gradient = ", grad)
    t = np.linspace(0, 5, 2)
   
    while grad > delta :
        for i in range (len(x)):
            gradLosw = (x[i][2] - np.dot(np.transpose(w), np.array([x[i][0], x[i][1], 1])))
            if(gradLosw != 0):
                #alpha = alphaSearch(w, x)
                w = w +  alpha * gradLosw * (np.array([x[i][0], x[i][1], 1]))
                compteur+=1
        grad =  gradient(w , x)
        #print(grad)
        #-------Plot de graphe----------------------------------------#
        if compteur % visualise == 0 :
            plt.clf() #clear figure
            #plt.title('graphe des points')
            plt.grid(False)#plot a grid
            plt.xlim(2, 5)
            plt.ylim(0, 3)
           
            if ( w[1]!= 0 ):
                yt = (-w[0]/w[1])*t - w[2]/w[1]
            # else : 
            #     yt = (-x[i][2]/w[0])*t                 
                plt.plot(t, yt, color  = col)
            #-----------------plot des points vert------------------------#
            plt.scatter(df['sepal_width'], df['petal_width'], c =target)

            plt.show()
            plt.pause(0.05)
    print('nomber d itiration : ', compteur)
    return w
       

#-----------------------------Algorithme One Vs One---------------------------#
def algorithmOvO(x):
    w1 = np.array([0, 0, 1])
    w2 = np.array([0, 0, 1])
    w3 = np.array([0, 0, 1])
    classe1 = []
    classe2 = []
    classe3 = []
    for i in range (len(x)):
        if(x[i][2]== 1 or x[i][2]== 2):
            if(x[i][2]== 1): #y = 1 en le remplace par -1
               classe1.append([x[i][0], x[i][1], -1])
            else: #c-a-d y = 2 en le remplase par 1
               classe1.append([x[i][0], x[i][1], 1])
               
        if(x[i][2]== 1 or x[i][2]== 3):
            if(x[i][2]== 1): #y = 1 en le remplace par -1
               classe2.append([x[i][0], x[i][1], -1])
            else: #c-a-d y = 3 en le remplase par 1
               classe2.append([x[i][0], x[i][1], 1])
               
        if(x[i][2]== 2 or x[i][2]== 3):
            if(x[i][2]== 2): #y = 2 en le remplace par -1
               classe3.append([x[i][0], x[i][1], -1])
            else: #c-a-d y = 3 en le remplase par 1
               classe3.append([x[i][0], x[i][1], 1])
         
    teta1 = AdalinOneVsOne(w1, classe1, 0.001, 'blue', 0.5, 100)
    teta2 = AdalinOneVsOne(w2, classe2, 0.001, 'red', 0.08, 1000)
    teta3 = AdalinOneVsOne(w3, classe3,  0.0001, 'green', 0.2, 10000)
    # # print('teta1 = ', teta1, '\nteta2 = ',teta2,'\nteta3 = ', teta3)
    
    plt.title('les beste separateurs ')
    plt.grid(False)#plot a grid
    plt.xlim(2, 4.5)
    plt.ylim(0, 2.5)
    #-----------------plot des points vert------------------------#
    plt.scatter(df['sepal_width'], df['petal_width'], c =target)

    #---------------plot separateur 1 -----------------#
    t = np.linspace(1, 4, 2)
    if ( teta1[1]!= 0 ):
        y1 = (-teta1[0]/teta1[1])*t - teta1[2]/teta1[1]        
    else : 
        y1 = (-x[i][2]/teta1[0])*t
    #---------------plot 2 separateur 2----------------#
    if ( teta2[1]!= 0 ):
        y2 = (-teta2[0]/teta2[1])*t - teta2[2]/teta2[1]        
    else : 
        y2 = (-x[i][2]/teta2[0])*t
    #---------------plot 3 separateur 3-----------------#
    if ( teta3[1]!= 0 ):
        y3 = (-teta3[0]/teta3[1])*t - teta3[2]/teta3[1]
    else : 
        y3 = (-x[i][2]/teta3[0])*t 
               
    #-----------le plote des separateur finale---------------------------------#
    plt.plot(t, y1, color='blue', label ='Separ C1 et C2')
    plt.plot(t, y2, color='red',  label ='Separ C1 et C3')
    plt.plot(t, y3, color='green',label ='Separ C2 et C3')

    
    plt.legend(loc ="lower right")   
    plt.show()



#------------------------teste de l algorithme--------------------------------#
algorithmOvO(data)







