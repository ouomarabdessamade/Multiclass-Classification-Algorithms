import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



#importation de data
dataframe = pd.read_csv('SegmentationOneVsOne.csv')
#print(data)



#la calcule de nobre des catégorie (nombre des classes)
dataframe['Segment'].value_counts(normalize = True)

#-----------------les caracteristiques :--------------------#
x = dataframe[['Recency', 'Tenure']]

#---la standarisation de data----------#
sc = StandardScaler()
X = sc.fit_transform(x)


#-----------transformer la colonne target en des 1, 2, 3-----#
target_value = { 'Potential': 1, 'Fence Sitter': 2, 'Loyal' : 3}
y = dataframe['Segment'] #la colonne Segment est notre vriable cible

#on remplace chaque categories par un nomber (1 , 2 et 3)
y = y.map(target_value)
y = np.array([y])
y_plot = y
#print(y)




#-----------la visualisation  de data
#plt.scatter(X[:,0], X[:,1], s=40, c=y_plot)


#--------------data----------------#
y = y.T
data = np.hstack((X, y))
#print(data)


t = np.linspace(-1.5, 2.5, 2)
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
            plt.xlim(-1.2, 2.5)
            plt.ylim(-1.7, 2)
           
            if ( w[1]!= 0 ):
                yt = (-w[0]/w[1])*t - w[2]/w[1]                 
                plt.plot(t, yt, color  = col)
            #-----------------plot des points vert------------------------#
            plt.scatter(X[:,0], X[:,1], s=40, c=y_plot)
            plt.pause(0.5)
            plt.clf() 
            plt.show()
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
         
    teta1 = AdalinOneVsOne(w1, classe1, 0.0001, 'blue', 0.02, 10)
    teta2 = AdalinOneVsOne(w2, classe2, 0.0001, 'red', 0.08, 10)
    teta3 = AdalinOneVsOne(w3, classe3,  0.0001, 'green', 0.2, 10)
    #print('teta1 = ', teta1, '\nteta2 = ',teta2,'\nteta3 = ', teta3)
    
    plt.title('les beste separateurs : One Vs One Adaline')
    plt.grid(False)#plot a grid
    plt.xlim(-1.2, 2.5)
    plt.ylim(-1.7, 2)
    #-----------------plot des points vert------------------------#
    plt.scatter(X[:,0], X[:,1], s=40, c=y_plot)

    #---------------plot separateur 1 -----------------#
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
