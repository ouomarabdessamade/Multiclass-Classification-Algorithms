import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#--data :
S = [
     [1, 1, 1, 1], [1, 2, 2, 1], [1, 3, 3, 1], [2, 1, 4, 1], [2, 3, 5, 1], [1, 5, 1, 1], [3, 5, 1, 7], 
     [4, 1, 1, 1], [4, 4, 3, 1], [2, 5, 1, 1], [3, 2, 2, 1], [3, 3, 3, 1], [5, 2, 4, 1], [6, 1, 5, 1], 
     [3, 1, 1, 1], [2, 2, 4, 1], [4, 2, 5, 1], [4, 3, 5, 1], [1, 4, 1, 1], [1, 6, 2, 1], [2, 6, 3, 1],
     [3, 6, 4, 1], [4, 6, 5, 1], [4, 5, 4, 1], [5, 4, 4, 1], [3, 4, 3, 1], [2, 4, 1, 1], [5, 3, 1, 1],
     [5, 1, 2, 1], [5, 3, 3, 1], [5, 5, 4, 1], [5, 6, 5, 1], [6, 2, 2, 1], [6, 3, 2, 1], [6, 5, 2, 1],
     [6, 4, 1, 1], [1, 7, 2, 1], [7, 5, 3, 1], [4, 7, 4, 1],  
     
     [11, 5, 7, 2], [12, 4, 8, 2], [12, 7, 9, 2], [13, 6, 10, 2], [9, 7,   6, 2], [10, 8, 7, 2], [10, 9, 8, 2],
     [11, 8, 9, 2], [13, 9,11, 2], [11, 9, 7, 2], [11, 11, 9, 2], [14, 8, 10, 2], [9, 8, 9,  2], [14, 9,8, 2 ],
     [8,  2, 7, 2], [8,  3, 9, 2], [9,  3,10, 2], [12, 2, 11, 2], [10, 3, 12, 2], [14, 4, 13,2], [8, 4, 14, 2],
     [12, 8, 7, 2], [10, 5, 8, 2], [12, 12,10,2], [13, 11,10, 2], [11, 12, 7, 2], [12, 14, 8,2], [9, 13, 9, 2],
     [10, 2, 6, 2], [9, 2, 7,  2], [9,  4,6,  2], [9, 5, 9,   2], [10, 4, 7,  2], [10, 6, 11,2], [11, 2, 9, 2],
     [13, 2, 7, 2], [14, 2, 7, 2], [14, 1, 6, 2], [13, 1, 9,  2], [11, 14, 10,2], [13, 5, 11,2], [10, 10, 9,2],
     [10, 11, 7,2], [12, 9, 10, 2], [12, 10,7, 2], [12, 11, 6, 2], [11, 7, 8,  2], [14, 7, 7, 2], [13, 13,8,2],
     [14, 12, 11,2],

     [5, 10, 14, 3], [1, 11, 15, 3], [1, 12,16, 3], [2, 9,17,  3], [2, 10, 18, 3], [1, 9, 19, 3], [2, 11, 16,3], 
     [4, 12, 14, 3], [4, 14, 15, 3], [4, 9, 16, 3], [4, 10,17, 3], [3, 9, 18,  3], [5, 11, 19,3], [6, 14, 16,3],
     [1, 10, 14, 3], [1, 13, 15, 3], [1, 14,16, 3], [2, 12,17, 3], [2, 13, 18, 3], [2, 14, 19,3], [3, 10, 17,3],
     [3, 11, 14, 3], [3, 12, 15, 3], [3, 14,16, 3], [4, 11,17, 3], [5, 13, 18, 3], [6, 12, 19,3], [5, 12, 18,3], 
     [6,  9, 14, 3], [6, 8,  15, 3], [6, 11,16, 3], [9, 14,17, 3], [7, 11, 18, 3], [7, 12, 19,3], [8, 13, 14,3], 
     [7, 13, 14, 3], [7, 14, 15, 3], [8, 14,16, 3], [3, 13,17, 3], [4, 13, 18, 3]
     ]




#--------------figure 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x1 = np.linspace(-1, 20, 1000)
x2 = np.linspace(-1, 20, 1000)
x1,x2 = np.meshgrid(x1, x2)

plt.title('Best separator 3D')

for i in range(len(S)):
    if S[i][3]==1:
        ax.scatter(S[i][0], S[i][1], S[i][2], alpha=1, color='red', marker='o', label='Classe1')


for i in range(len(S)):
    if S[i][3]==2:
        ax.scatter(S[i][0], S[i][1], S[i][2], alpha=1, color='green', marker='o', label='Classe2')

for i in range(len(S)):
    if S[i][3]==3:
        ax.scatter(S[i][0], S[i][1], S[i][2], alpha=1, color='blue', marker='o', label='Classe3')





#-----------------------------loss function-----------------------------------#
def lossfunction(poid, x):
    som = 0
    for i in range(len(x)):
        if ( np.sign(np.dot(np.transpose(poid), np.array([x[i][0], x[i][1], x[i][2], 1])))  != x[i][3] ):
            som = som + 1
    return som /len(x)



def perceptron(w0, S):
    w = w0
    compteur = 0
    lossfon = lossfunction(w, S)
    while lossfon!= 0 :
        for i in range(len(S)):
            if np.sign(np.dot(np.transpose(w),np.array([S[i][0], S[i][1], S[i][2], 1]))) != S[i][3] :
                w = w + S[i][3] * (np.array([S[i][0],S[i][1], S[i][2], 1])) 
                compteur = compteur+1
            #end if
        #end for
        lossfon = lossfunction(w, S)
    return w


#-----------------------------Algorithme-----------------------------------#
def algorithmOvO(w1, w2, w3, x):
    classe1 = []
    classe2 = []
    classe3 = []
    for i in range (len(x)):
        if(x[i][3]== 1 or x[i][3]== 2):
            if(x[i][3]== 1): #y = 1 en le remplace par -1
               classe1.append([x[i][0], x[i][1], x[i][2], -1])
            else: #c-a-d y = 2 en le remplase par 1
               classe1.append([x[i][0], x[i][1], x[i][2], 1])
               
        if(x[i][3]== 1 or x[i][3]== 3):
            if(x[i][3]== 1): #y = 1 en le remplace par -1
               classe2.append([x[i][0], x[i][1], x[i][2], -1])
            else: #c-a-d y = 3 en le remplase par 1
               classe2.append([x[i][0], x[i][1], x[i][2], 1])
               
        if(x[i][3]== 2 or x[i][3]== 3):
            if(x[i][3]== 2): #y = 2 en le remplace par -1
               classe3.append([x[i][0], x[i][1], x[i][2], -1])
            else: #c-a-d y = 3 en le remplase par 1
               classe3.append([x[i][0], x[i][1], x[i][2], 1])
               
    teta1 = perceptron(w1, classe1)
    teta2 = perceptron(w2, classe2)
    teta3 = perceptron(w3, classe3)
    #print('teta 1 = ', teta1, 'teta 2 = ', teta2, 'teta 3 = ', teta3)
    return teta1, teta2, teta3
    



w1 = np.array([1, 1, 1, 1])
w2 = np.array([0, 0, 1, 1])
w3 = np.array([0, 0, 1, 1])

theta1, theta2, theta3 = algorithmOvO(w1, w2, w3, S)

#-------Plot de graphe Resultat final-----------------------------------------#
ax.set_xlim3d(-1, 20)
ax.set_ylim3d(-1, 20)
ax.set_zlim3d(-1, 20)


#---equation de separateur1------------------------------------#
if ( theta1[2]!= 0 ):
      x3= (-theta1[0]*x1 - theta1[1]*x2 - theta1[3])/theta1[2]
      ax.plot_surface(x1, x2, x3, color='green', alpha= 0.2, label ='Separ C1 et C2')
     
elif theta1[2]==0 and theta1[1]!=0 :
      x3= (-theta1[0]*x1 - theta1[3]*x2) / theta1[1]
      ax.plot_surface(x1, x2, x3, color='green', alpha= 0.2, label ='Separ C1 et C2')
else : 
      x3= (-theta1[0]*x1 - theta1[1]*x2 - theta1[2])/theta1[1]
      ax.plot_surface(x1, x2, x3, color='green', alpha= 0.2, label ='Separ C1 et C2')

#---equation de separateur 2------------------------------------#
if ( theta2[2]!= 0 ):
      x3= (-theta2[0]*x1 - theta2[1]*x2 - theta2[3])/theta2[2]
      ax.plot_surface(x1, x2, x3, color='red', alpha= 0.2, label ='Separ C1 et C3')
     
elif theta2[2]==0 and theta2[1]!=0 :
      x3= (-theta2[0]*x1 - theta2[3]*x2) / theta2[1]
      ax.plot_surface(x1, x2, x3, color='green', alpha= 0.2, label ='Separ C1 et C3')
else : 
      x3= (-theta2[0]*x1 - theta2[1]*x2 - theta2[2])/theta2[1]
      ax.plot_surface(x1, x2, x3, color='green', alpha= 0.2, label ='Separ C1 et C3')

#---equation de separateur 3------------------------------------#
if ( theta3[2]!= 0 ):
      x3= (-theta3[0]*x1 - theta3[1]*x2 - theta3[3])/theta3[2]
      ax.plot_surface(x1, x2, x3, color='blue', alpha= 0.2, label ='Separ C2 et C3')
     
elif theta3[2]==0 and theta3[1]!=0 :
      x3= (-theta3[0]*x1 - theta3[3]*x2) / theta3[1]
      ax.plot_surface(x1, x2, x3, color='green', alpha= 0.2, label ='Separ C2 et C3')
else : 
      x3= (-theta3[0]*x1 - theta3[1]*x2 - theta3[2])/theta3[1]
      ax.plot_surface(x1, x2, x3, color='green', alpha= 0.2, label ='Separ C2 et C3')

plt.show()
 
#plt.pause(20)









