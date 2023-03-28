import pandas as pd
import numpy as np
#import os
import datetime as dtku
#import matplotlib
from scipy import spatial

from skimage.feature import greycomatrix, greycoprops
from skimage import data

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import math

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from numpy import array


############################### HAVERSINE FOR NEAREST ROADS

def date_convert(date_to_convert):
     return dtku.datetime.strptime(date_to_convert, '%H:%M:%S').time()



####################################################################################
    
####################################################################################
########################################## viterbi algorithm
def viterbi(pi, a, b, obs):
    
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]
    
    # init blank path
    path = path = np.zeros(T,dtype=int)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))
    
    # init delta and phi 
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    #print('\nStart Walk Forward\n')    #turn of comment
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]] 
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            #print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t])) #turn of comment
    
    # find optimal path
    #print('-'*50) #turn of comment
    #print('Start Backtrace\n') #turn of comment
    path[T-1] = np.argmax(delta[:, T-1])
    for t in range(T-2, -1, -1):
        path[t] = phi[path[t+1], [t+1]]
        #print('path[{}] = {}'.format(t, path[t])) #turn of comment
        
    return path, delta, phi
####################################################################################


#######################################################################
#######################################################################
#Split 20 %
peRatus = 0.2

#######################################################################################################
#######################################################################################################
#   ENTRY PROBABILITY 
#   myDataPemerhatian (observation state)
#   myInisialMarkov   (pi)
#   myMarkov          (Markov Chain)
#######################################################################################################
oSatu = pd.DataFrame({'Index':['Clear','Congestion'],
        'Time 1': [0.17537171178040412, 0.07782101167315175],
        'Time 2': [0.11742279832253145, 0.669260700389105],
        'Time 3': [0.16240945482272207 , 0.21011673151750973],
        'Time 4': [0.18299656881433474, 0],
        'Time 5': [0.17918414029736943, 0.038910505836575876],
        'Time 6': [0.18261532596263821, 0.0038910505836575876],
          })
oSatu = oSatu.set_index('Index')


oDua = pd.DataFrame({'Index':['Clear','Congestion'],
        'Time 1': [0.18294573643410852, 0.02666666666666667],
        'Time 2': [0.12170542635658915, 0.5533333333333333],
        'Time 3': [0.14224806201550388, 0.37666666666666665],
        'Time 4': [0.18604651162790697, 0],
        'Time 5': [0.1810077519379845, 0.043333333333333335],
        'Time 6': [0.18604651162790697, 0],
          })
oDua = oDua.set_index('Index')


oTiga = pd.DataFrame({'Index':['Clear','Congestion'],
        'Time 1': [0.1837754709727028, 0.007168458781362007],
        'Time 2': [0.12379853902345252, 0.5663082437275986],
        'Time 3': [0.1437908496732026 , 0.37992831541218636],
        'Time 4': [0.1845444059976932, 0],
        'Time 5': [0.17954632833525566, 0.04659498207885305],
        'Time 6': [0.1845444059976932, 0],
          })
oTiga = oTiga.set_index('Index')

oEmpat = pd.DataFrame({'Index':['Clear','Congestion'],
        'Time 1': [0.18258859784283513, 0.02112676056338028],
        'Time 2': [0.12249614791987673, 0.5704225352112676],
        'Time 3': [0.1448382126348228 , 0.36619718309859156],
        'Time 4': [0.18489984591679506, 0],
        'Time 5': [0.18258859784283513, 0.02112676056338028],
        'Time 6': [0.18258859784283513, 0.02112676056338028],
          })
oEmpat = oEmpat.set_index('Index')

myDataPemerhatian = {}
myDataPemerhatian[0] = oSatu
myDataPemerhatian[1] = oDua
myDataPemerhatian[2] = oTiga
myDataPemerhatian[3] = oEmpat



myInisialMarkov = {0: array([0.91076389, 0.08923611],float), 1: array([0.89583333, 0.10416667],float), 2: array([0.903125, 0.096875],float), 3: array([0.90138889, 0.09861111],float)}

myMarkov = {}
myMarkov[0] = array([[0.96338673, 0.03661327],[0.37354086, 0.62645914]],float)
myMarkov[1] = array([[0.96161303, 0.03838697],[0.33      , 0.67      ]],float)
myMarkov[2] = array([[0.96269231, 0.03730769],[0.34767025, 0.65232975]],float)
myMarkov[3] = array([[0.96647399, 0.03352601],[0.30633803, 0.69366197]],float)

####################################################################################
############################## HITUNG HMM
#BUKA DATA SET SPLIT 20 %
##################################################################
#### buka dataset .csv
mDataKu =  pd.read_csv("C://Users/acer/Documents/myDataSet.csv")
mDataKu['TIMESTAMP'] = mDataKu['TIMESTAMP'].apply(date_convert)
mDataKu = mDataKu.set_index('TIMESTAMP')


X_train, X_test, y_train, y_test = train_test_split(mDataKu, mDataKu, test_size=peRatus, shuffle = False)

tSatu =dtku.datetime.strptime('05:00', '%H:%M').time()

tDua = dtku.datetime.strptime('17:00', '%H:%M').time()
# ###############################################################################
y_test= y_test[(y_test.index >= tSatu)& (y_test.index<=tDua)]

###############################################################################
###############################################################################
########################## 
states = ['Time 1', 'Time 2', 'Time 3', 'Time 4', 'Time 5', 'Time 6']
hidden_states = ['Clear', 'Congestion']
observable_states = states
c_df = pd.DataFrame(columns=observable_states, index=hidden_states)
c_df.loc[hidden_states[0]] = [ 0, 0, 0, 0, 0, 0]
c_df.loc[hidden_states[1]] = [ 0, 0, 0, 0, 0, 0]


obs = y_test.loc[:,'TIME_CLUSTER'].values
obs = obs.astype(int)


hasil_result_time = {}

#obs_map = {'Different Time':0, 'On Time':1}
obs_map = {'Time 1':0, 'Time 2':1, 'Time 3':2, 'Time 4':3, 'Time 5':4, 'Time 6':5}
inv_obs_map = dict((v,k) for k, v in obs_map.items())
obs_seq = {}


j=0

for j in range(1, len(myDataPemerhatian)): 
    #obs = np.array([2,2,1,5,0,0,1,3])
    #obs = obs.astype(int)
    obs_seq[j] = [inv_obs_map[v] for v in list(obs)]
    
    pi = [myInisialMarkov[j].item(0),myInisialMarkov[j].item(1)] # initial probabilities vector

    #pi = [myInisialMarkov[0].item(0), myInisialMarkov[0].item(1)]
    state_space = pd.Series(pi, index=hidden_states, name='states')
    a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)

    for i in range(0,2):
        a_df.loc[hidden_states[i]] = [myMarkov[j].item(i,0), myMarkov[j].item(i,1)]
    a = a_df.values

    observable_states = states
    b_df = myDataPemerhatian[j]  
    b = b_df.values
    path, delta, phi = viterbi(pi, a, b, obs)
    state_map = {0:'Clear', 1:'Congestion'}
    state_path = [state_map[v] for v in path]
    hasil_result_time[j] = pd.DataFrame().assign(Observation=obs_seq[j]).assign(Best_Path=state_path)
    


a_test = y_test.reset_index(drop=True)

hasil_result_time[1]

i=0
for i in range(1,len(myDataPemerhatian)):
    hasil_result_time[i]['ASAL']= a_test.iloc[:,i].copy()
    hasil_result_time[i].loc[hasil_result_time[i]['Best_Path'] == 'Congestion', 'HASIL'] = 1
    hasil_result_time[i].loc[hasil_result_time[i]['Best_Path'] == 'Clear', 'HASIL'] = 0
    


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

apaya = {}
con_matrix_aja = {}
hasil_akurasi = pd.DataFrame(columns=['akurasi', 'presisi', 'recall', 'f1'])
for i in range(1, len(myDataPemerhatian)):
    con_matrix_aja[i] = confusion_matrix(hasil_result_time[i].loc[:,'ASAL'], hasil_result_time[i].loc[:,'HASIL'])
    apaya[i] = classification_report(hasil_result_time[i].loc[:,'ASAL'], hasil_result_time[i].loc[:,'HASIL'])
    hasil_akurasi.loc[i,'akurasi'] = metrics.accuracy_score(hasil_result_time[i].loc[:,'ASAL'], hasil_result_time[i].loc[:,'HASIL'])
    hasil_akurasi.loc[i,'presisi']= metrics.precision_score(hasil_result_time[i].loc[:,'ASAL'], hasil_result_time[i].loc[:,'HASIL'], average='binary')
    hasil_akurasi.loc[i,'recall'] = metrics.recall_score(hasil_result_time[i].loc[:,'ASAL'], hasil_result_time[i].loc[:,'HASIL'], average='binary')
    hasil_akurasi.loc[i,'f1']= metrics.f1_score(hasil_result_time[i].loc[:,'ASAL'], hasil_result_time[i].loc[:,'HASIL'], average='binary')

