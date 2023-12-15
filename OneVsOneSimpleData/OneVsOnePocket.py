import numpy as np
import matplotlib.pyplot as plt




#---------------------------data-----------------------------------------------#
data = [
     [1, 1, 1], [1, 2, 1], [1, 3, 1], [2, 1, 1], [2, 3, 1], [1, 5, 1], [3, 5, 1], 
     [4, 1, 1], [4, 4, 1], [2, 5, 1], [3, 2, 1], [3, 3, 1], [5, 2, 1], [6, 1, 1], 
     [3, 1, 1], [2, 2, 1], [4, 2, 1], [4, 3, 1], [1, 4, 1], [1, 6, 1], [2, 6, 1],
     [3, 6, 1], [4, 6, 1], [4, 5, 1], [5, 4, 1], [3, 4, 1], [2, 4, 1], [5, 3, 1],
     [5, 1, 1], [5, 3, 1], [5, 5, 1], [5, 6, 1], [6, 2, 1], [6, 3, 1], [6, 5, 1],
     [6, 4, 1], [1, 7, 1], [7, 5, 1], [4, 7, 1],  
     
     [11, 5,  2], [12, 4,  2], [12, 7,  2], [13, 6,  2], [9, 7,  2], [10, 8, 2], [10, 9,  2],
     [11, 8,  2], [13, 9,  2], [11, 9,  2], [11, 11, 2], [14, 8, 2], [9, 8,  2], [14, 9,  2],
     [8, 2,   2], [8, 3,   2], [9,  3,  2], [12, 2,  2], [10, 3, 2], [14, 4, 2], [8, 4,   2],
     [12, 8,  2], [10, 5, 2 ], [12, 12, 2], [13, 11, 2], [11, 12, 2],[12, 14, 2],[9, 13,  2],
     [10, 2,  2], [9, 2,   2], [9,  4,  2], [9, 5,   2], [10, 4,  2],[10, 6,  2],[11, 2,  2],
     [13, 2,  2], [14, 2,  2], [14, 1,  2], [13, 1,  2], [11, 14, 2],[13, 5,  2],[10, 10, 2],
     [10, 11, 2], [12, 9,  2], [12, 10, 2], [12, 11, 2], [11, 7,  2],[14, 7,  2],[13, 13, 2],
     [14, 12, 2],

     [5, 10,  3], [1, 11,  3], [1, 12,  3], [2, 9,   3], [2, 10, 3], [1, 9,  3], [2, 11,  3], 
     [4, 12,  3], [4, 14,  3], [4, 9,   3], [4, 10,  3], [3, 9,  3], [5, 11, 3], [6, 14,  3],
     [1, 10,  3], [1, 13,  3], [1, 14,  3], [2, 12,  3], [2, 13, 3], [2, 14, 3], [3, 10,  3],
     [3, 11,  3], [3, 12,  3], [3, 14,  3], [4, 11,  3], [5, 13, 3], [6, 12, 3], [5, 12,  3], 
     [6,  9,  3], [6, 8,   3], [6, 11,  3], [9, 14,  3], [7, 11,  3],[7, 12, 3], [8, 13, 3], 
     [7, 13,  3], [7, 14,  3], [8, 14,  3], [3, 13,  3], [4, 13, 3]
     ]

#---------------data sous dictionaire juste pour le plot----------------------#
data_dict = {
    #'keys' : 'values',
    
    #-----classe 1
    '1, 1':'1', '1, 2':'1', '1, 3':'1', '2, 1':'1', '2, 3':'1', '1, 5':'1', '3, 5':'1', 
    '4, 1':'1', '4, 4':'1', '2, 5':'1', '3, 2':'1', '3, 3':'1', '5, 2':'1', '6, 1':'1', 
    '3, 1':'1', '2, 2':'1', '4, 2':'1', '4, 3':'1', '1, 4':'1', '1, 6':'1', '2, 6':'1', 
    '3, 6':'1', '4, 6':'1', '4, 5':'1', '5, 4':'1', '3, 4':'1', '2, 4':'1', '5, 3':'1', 
    '5, 1':'1', '5, 3':'1', '5, 5':'1', '5, 6':'1', '6, 2':'1', '6, 3':'1', '6, 5':'1',
    '6, 4':'1', '1, 7':'1', '7, 5':'1', '4, 7':'1',  
    #-----classe 2
    '11, 5':'2', '12, 4':'2', '12, 7':'2' , '13, 6' :'2', '9, 7'  :'2', '10, 8':'2', '10, 9':'2',
    '11, 8':'2', '13, 9':'2', '11, 9':'2' , '11, 11':'2', '14, 8' :'2', '9, 8' :'2', '14, 9':'2',
    '8, 2' :'2', '8, 3' :'2', '9, 3' :'2' , '12, 2' :'2', '10, 3' :'2', '14, 4':'2', '8, 4' :'2',
    '12, 8':'2', '10, 5':'2', '12,12': '2', '13, 11':'2', '11, 12':'2', '12,14':'2', '9, 13':'2',
    '10, 2':'2', '9,  2':'2',  '9, 4': '2', '9,  5' :'2', '10, 4' :'2', '10, 6':'2', '11, 2':'2',
    '13, 2':'2', '14, 2':'2', '14, 1': '2', '13, 1' :'2', '11, 14':'2', '13, 5':'2', '10, 10':'2',
    '10, 11':'2', '12, 9':'2', '12, 10':'2', '12, 11':'2', '11, 7':'2', '14, 7':'2', '13, 13':'2',
    '14, 12':'2',

    #-----classe 3
    '5, 10':'3', '1, 11':'3', '1, 12':'3', '2, 9' :'3', '2, 10':'3', '1, 9' :'3', '2, 11':'3', 
    '4, 12':'3', '4, 14':'3', '4, 9' :'3', '4, 10':'3', '3, 9' :'3', '5, 11':'3', '6, 14':'3',
    '1, 10':'3', '1, 13':'3', '1, 14':'3', '2, 12':'3', '2, 13':'3', '2, 14':'3', '3, 10':'3',
    '3, 11':'3', '3, 12':'3', '3, 14':'3', '4, 11':'3', '5, 13':'3', '6, 12':'3', '5, 12':'3',
    '6, 9' :'3', '6, 10':'3', '6, 11':'3', '9, 14':'3', '7, 11':'3', '7, 12':'3', '8, 13':'3', 
    '7, 13':'3', '7, 14':'3', '8, 14':'3', '3, 13':'3', '4, 13':'3'
}

#---------------------fonction pour visualise---------------------------------#
def get_point(data, label):
    x_coords = [float(point.split(",")[0])for point in data.keys() if data[point] == label]
    y_coords = [float(point.split(",")[1])for point in data.keys() if data[point] == label]
    return x_coords, y_coords


#---------------------la visualisation de data--------------------------------#plt.ion()
'''
plt.title('graphe des points')
plt.grid(False)#plot a grid
plt.xlim(0, 15)
plt.ylim(0, 15)

#-----------------plot des points vert------------------------#
x_coords, y_coords = get_point(data_dict, '1')
plt.plot(x_coords, y_coords, 'gx')
#---------------plot des points rouge-------------------------#
x_coords, y_coords = get_point(data_dict, '2')
plt.plot(x_coords, y_coords, 'r*')
#---------------plot des points blue-------------------------#
x_coords, y_coords = get_point(data_dict, '3')
plt.plot(x_coords, y_coords, 'b+')

plt.show()
'''


#-----------------------------loss function-----------------------------------#
def lossfunction(poid, x):
    som = 0
    for i in range(len(x)):
        if ( np.sign(np.dot(np.transpose(poid), np.array([x[i][0], x[i][1], 1])))  != x[i][2] ):
            som = som + 1
    
    return som /len(x)

  
#------------------------Algorithme de Pocket----------------------------------#
def PocketOneVsOne(w0, x, alpha, col, t_max = 200):
    plt.ion()
    compteur = 0
    ws = w0
    t = np.linspace(0, 15, 2)
    for k in range(len(x)):
        if np.sign(np.dot(np.transpose(ws),np.array([x[k][0],x[k][1], 1]))) != x[k][2] :
            ws = ws + x[k][2] * (np.array([x[k][0], x[k][1], 1])) 
           
    lossfuncWs = lossfunction(ws, x)

    wt = [0, 0, 1]
    for i in range(0, t_max):
        for j in range(len(x)):
            if np.sign(np.dot(np.transpose(wt), np.array([x[j][0],x[j][1], 1]))) != x[j][2] :
                wt = wt + x[j][2] * (np.array([x[j][0], x[j][1], 1])) 
                compteur+=1
        #----evaluation de Ls(w(t)) 
        lsWt = lossfunction(wt, x)
        
        if(lsWt < lossfuncWs):
            ws = wt
            lossfuncWs = lossfunction(ws, x)
        #-------Plot de graphe------------------------------------------------#
            plt.clf() #clear figure
            #plt.title('graphe des points')
            plt.grid(False)#plot a grid
            plt.xlim(0, 16)
            plt.ylim(0, 16)
           
            if ( ws[1]!= 0 ):
                yt = (-ws[0]/ws[1])*t - ws[2]/ws[1]
            # else : 
            #     yt = (-x[i][2]/w[0])*t                 
                plt.plot(t, yt, color  = col)
           
            #-----------------plot des points vert------------------------#
            x_coords, y_coords = get_point(data_dict, '1')
            plt.plot(x_coords, y_coords, 'gx')
            #---------------plot des points rouge-------------------------#
            x_coords, y_coords = get_point(data_dict, '2')
            plt.plot(x_coords, y_coords, 'r*')
            #---------------plot des points blue-------------------------#
            x_coords, y_coords = get_point(data_dict, '3')
            plt.plot(x_coords, y_coords, 'b+')
    
            plt.pause(0.05)
            plt.show()
    #print('le nomber d itiration = ', compteur)
    return ws
       

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
         
    teta1 = PocketOneVsOne(w1, classe1, 0.0001, 'blue')
    teta2 = PocketOneVsOne(w2, classe2, 0.0001, 'red')
    teta3 = PocketOneVsOne(w3, classe3, 0.00001, 'green')
    # print('teta1 = ', teta1, '\nteta2 = ',teta2,'\nteta3 = ', teta3)
    
    plt.title('Pocket One Vs One : beste separateurs  ')
    plt.grid(False)#plot a grid
    plt.xlim(0, 15)
    plt.ylim(0, 15)
    #-----------------plot des points vert------------------------#
    x_coords, y_coords = get_point(data_dict, '1')
    plt.plot(x_coords, y_coords, 'gx', label='classe 1')

    #---------------plot des points rouge-------------------------#
    x_coords, y_coords = get_point(data_dict, '2')
    plt.plot(x_coords, y_coords, 'r*', label='classe 2')

    #---------------plot des points blue-------------------------#
    x_coords, y_coords = get_point(data_dict, '3')
    plt.plot(x_coords, y_coords, 'b+', label='classe 3')

    #---------------plot separateur 1 -----------------#
    t = np.linspace(0, 15, 2)
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
        #print(y3)
    else : 
        y3 = (-x[i][2]/teta3[0])*t 
               
    #-----------le plote des separateur finale---------------------------------#
    plt.plot(t, y1,color='blue', label ='Separ C1 et C2')
    plt.plot(t, y2,color='red',  label ='Separ C1 et C3')
    plt.plot(t, y3,color='green',label ='Separ C2 et C3')

    
    plt.legend(loc ="lower right")   
    plt.show()



#------------------------teste de l algorithme--------------------------------#
algorithmOvO(data)





