# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import math,random,os
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)

exp=2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274274663919320030599218174135966290435729003342952605956307381323286279434907632338298807531952510190115738341879307021540891499348841675092447614606680822648001684774118537423454424371075390777449920695517027618386062613313845830007520449338265602976067371132007093287091274437470472306969772093101416928368190255151086574637721112523897844250569536967707854499699679468644549059879316368892300987931277361782154249992295763514822082698951936680331825288693984964651058209392398294887933203625094431173012381970684161403970198376793206832823764648042953118023287825098194558153017567173613320698112509961818815930416903515988885193458072738667385894228792284998920868058257492796104841984443634632449684875602336248270419786232090021609902353043699418491463140934317381436405462531520961836908887070167683964243781405927145635490613031072085103837505101157477041718986106873969655212671546889570350354
c=	299792458 #M/S
seed=42
MeV_to_KG=5.60958*(10**29)
MeVs_to_kgms=5.344/(10**22)
electron_mass=0.0005489 #AEM
electron_mass_kg=9.1093837/(10**31) #9.1093837
aem=6.022e+26#1 кг
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(seed)


train = pd.read_excel('For_model_labled.xlsx')
test = pd.read_excel('For_check_unlabled.xlsx')
#full=pd.read_csv('dielectron.csv')

#full.columns = full.columns.str.replace('px1 ', 'px1')

#full=full[train.columns]
#train=full
#train=train.drop_duplicates()


#train["M"]=train["M"].fillna(train["M"].median())
#test["M"]=test["M"].fillna(test["M"].median())


#train['p_1']=abs((train['pt1']*(exp**(2*train['eta1']) +1))/(exp**(2*train['eta1'])-1))
#train['p_2']=abs((train['pt2']*(exp**(2*train['eta2']) +1))/(exp**(2*train['eta2'])-1))

#test['p_1']=abs((test['pt1']*(exp**(2*test['eta1']) +1))/(exp**(2*test['eta1'])-1))
#test['p_2']=abs((test['pt2']*(exp**(2*test['eta2']) +1))/(exp**(2*test['eta2'])-1))


def calc_M(phi1,phi2,eta1,eta2,p1,p2):
    return np.sqrt( 2*p1*p2* ( np.cosh(eta1-eta2) - np.cos( phi1 - phi2 ) ) )


train['M_new']=calc_M(train['phi1'],train['phi2'],train['eta1'],train['eta2'],train['pt1'],train['pt2'])#more accurate
test['M_new']=calc_M(test['phi1'],test['phi2'],test['eta1'],test['eta2'],test['pt1'],test['pt2'])#more accurate
train=train.drop('M',axis=1)
test=test.drop('M',axis=1)

#train['p1']=np.sqrt(((train['E1']*1000)**2)-(electron_mass**2)*(931.5**2))/931.5
#train['p2']=np.sqrt(((train['E2']*1000)**2)-(electron_mass**2)*(931.5**2))/931.5
#test['p1']=np.sqrt(((test['E1']*1000)**2)-(electron_mass**2)*(931.5**2))/931.5
#test['p2']=np.sqrt(((test['E2']*1000)**2)-(electron_mass**2)*(931.5**2))/931.5#proper

#train['E1']=np.sqrt((train['px1']**2)+(train['py1']**2)+(train['pz1']**2)) #more accurate
#train['E2']=np.sqrt((train['px2']**2)+(train['py2']**2)+(train['pz2']**2)) #more accurate

#test['E1']=np.sqrt((test['px1']**2)+(test['py1']**2)+(test['pz1']**2)) #more accurate
#test['E2']=np.sqrt((test['px2']**2)+(test['py2']**2)+(test['pz2']**2)) #more accurate
train.Q1.replace({-1:0,}, inplace=True)
test.Q1.replace({-1:0,}, inplace=True)
def get_joules(num):
    return 1.6022e-13*num

def get_kgms(num):
    return num*MeVs_to_kgms

'''train['m1_unstatic']=calc_unstatic_mass(train['E1'],train['p1'])
train['m2_unstatic']=calc_unstatic_mass(train['E2'],train['p2'])
test['m1_unstatic']=calc_unstatic_mass(test['E1'],test['p1'])
test['m2_unstatic']=calc_unstatic_mass(test['E2'],test['p2'])'''

'''train['pl1']=np.sqrt((train['py1']**2)+(train['pz1']**2))
train['pl2']=np.sqrt((train['py2']**2)+(train['pz2']**2))
test['pl1']=np.sqrt((test['py1']**2)+(test['pz1']**2))
test['pl2']=np.sqrt((test['py2']**2)+(test['pz2']**2))'''

train['m1']=(train['E1']*1000)/(931.5**2)
train['m2']=(train['E2']*1000)/(931.5**2)

test['m1']=(test['E1']*1000)/(931.5**2)
test['m2']=(test['E2']*1000)/(931.5**2)

'''train['rest_energy']=(train['M_new']*1000)*(931.5**2)
test['rest_energy']=(test['M_new']*1000)*(931.5**2)'''

def calc_speed(E):
    return c*(np.sqrt(1- ((electron_mass_kg*(c**2))/(E+electron_mass_kg*(c**2)))**2 ))

'''train['v1']=c-calc_speed(get_joules(train['E1']*1000))
train['v2']=c-calc_speed(get_joules(train['E2']*1000))

test['v1']=c-calc_speed(get_joules(test['E1']*1000))
test['v2']=c-calc_speed(get_joules(test['E2']*1000))

train['w1']=np.arctanh((c-train['v1'])/c)
train['w2']=np.arctanh((c-train['v2'])/c)
test['w1']=np.arctanh((c-test['v1'])/c)
test['w2']=np.arctanh((c-test['v2'])/c)
'''
def calc_momenta(v):
    return electron_mass_kg*v/np.sqrt(1-((v**2)/(c**2)))

'''train['momenta1_kgm_per_s']=calc_momenta(c-train['v1'])
train['momenta2_kgm_per_s']=calc_momenta(c-train['v2'])
test['momenta1_kgm_per_s']=calc_momenta(c-test['v1'])
test['momenta2_kgm_per_s']=calc_momenta(c-test['v2'])'''

#train=train.drop(columns=['v2'])
#test=test.drop(columns=['v2']) #low importance

def calc_rapidity(eta,transverse):
    return eta-(np.tanh(eta)/2)*(electron_mass_kg/transverse)**2

'''train['y1']=calc_rapidity(train['eta1'],get_joules(train['pt1']*1000))
train['y2']=calc_rapidity(train['eta2'],get_joules(train['pt2']*1000))
test['y1']=calc_rapidity(test['eta1'],get_joules(test['pt1']*1000))
test['y2']=calc_rapidity(test['eta2'],get_joules(test['pt2']*1000))

train['diff_R']=np.sqrt(((train['y1']-train['y2'])**2)+((train['phi1']-train['phi2'])**2))
test['diff_R']=np.sqrt(((test['y1']-test['y2'])**2)+((test['phi1']-test['phi2'])**2))

train['mt1']=np.sqrt((get_joules(train['pt1']*1000)**2)+electron_mass_kg**2)
train['mt2']=np.sqrt((get_joules(train['pt2']*1000)**2)+electron_mass_kg**2)
test['mt1']=np.sqrt((get_joules(test['pt1']*1000)**2)+electron_mass_kg**2)
test['mt2']=np.sqrt((get_joules(test['pt2']*1000)**2)+electron_mass_kg**2)'''

def calc_unstatic_mass(E,p):
    return np.sqrt( E**2 - ((p**2) * (c**2)) )/(c**2)

'''train['unstatic_mass_kg1']=calc_unstatic_mass(get_joules(train['E1']*1000),train['momenta1_kgm_per_s'])
train['unstatic_mass_kg2']=calc_unstatic_mass(get_joules(train['E2']*1000),train['momenta2_kgm_per_s'])
test['unstatic_mass_kg1']=calc_unstatic_mass(get_joules(test['E1']*1000),test['momenta1_kgm_per_s'])
test['unstatic_mass_kg2']=calc_unstatic_mass(get_joules(test['E2']*1000),test['momenta2_kgm_per_s'])'''

def get_mevs_mom(p):
    return p/MeVs_to_kgms
def get_kgs(m):
    return m*aem

'''train['momenta1_kgm_per_s']=get_mevs_mom(train['momenta1_kgm_per_s'])
train['momenta2_kgm_per_s']=get_mevs_mom(train['momenta2_kgm_per_s'])
test['momenta1_kgm_per_s']=get_mevs_mom(test['momenta1_kgm_per_s'])
test['momenta2_kgm_per_s']=get_mevs_mom(test['momenta2_kgm_per_s'])'''

'''train['unstatic_mass_kg1']=get_kgs(train['unstatic_mass_kg1'])
train['unstatic_mass_kg2']=get_kgs(train['unstatic_mass_kg2'])
test['unstatic_mass_kg1']=get_kgs(test['unstatic_mass_kg1'])
test['unstatic_mass_kg2']=get_kgs(test['unstatic_mass_kg2'])'''
print(train)
'''train=train.drop(columns=['momenta1_kgm_per_s','momenta2_kgm_per_s'])
test=test.drop(columns=['momenta1_kgm_per_s','momenta2_kgm_per_s'])'''

unimportant=['px1','px2','py1','py2','pz1','pz2','phi1','phi2']#,'w1','w2','y2']

train=train.drop(columns=unimportant)
test=test.drop(columns=unimportant)

high_corr=['m1','m2','y1','eta2','M_new'] #high corr with E1,E2,eta1,y2,rest_energy

#train=train.drop(columns=high_corr)
#test=test.drop(columns=high_corr)


print(test['Q1'].value_counts()/len(test))


#train=train.drop(columns=['Q1'])
#test=test.drop(columns=['Q1'])

'''copy_cols=['px1','py1','pz1','pt1',
           'phi1','px2','py2','pz2',
           'pt2','phi2','v1','v2',
           'w1','w2','diff_R']

for i in copy_cols:
    if i in list(train.columns):
        print(i)
        train[i+'_copy']=train[i]
        test[i + '_copy'] = test[i]'''

#plt.figure(figsize=(50, 50))
#sns.heatmap(train.corr(), annot=True,linewidths=1)
#plt.show()

y = train['Q2']
X = train.drop(columns = ['Q2'])

X=X.drop(columns=['Event','Run'])
ev=test['Event']
test=test.drop(columns=['Event','Run'])


X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = seed, test_size = 0.2)

from catboost import CatBoostClassifier
params_no_its={'nan_mode': 'Min', 'eval_metric': 'F1', 'sampling_frequency': 'PerTree', 'leaf_estimation_method': 'Newton', 'grow_policy': 'SymmetricTree', 'penalties_coefficient': 1, 'boosting_type': 'Plain', 'model_shrink_mode': 'Constant', 'feature_border_type': 'GreedyLogSum',   'l2_leaf_reg': 3, 'random_strength': 1, 'rsm': 1, 'boost_from_average': False, 'model_size_reg': 0.5,  'subsample': 0.800000011920929, 'use_best_model': True, 'class_names': [-1, 1], 'depth': 6, 'posterior_sampling': False, 'border_count': 254, 'classes_count': 0, 'auto_class_weights': 'None', 'sparse_features_conflict_fraction': 0, 'leaf_estimation_backtracking': 'AnyImprovement', 'best_model_min_trees': 1, 'model_shrink_rate': 0, 'min_data_in_leaf': 1, 'loss_function': 'Logloss', 'learning_rate': 0.18095199763774872, 'score_function': 'Cosine', 'task_type': 'CPU', 'leaf_estimation_iterations': 10, 'bootstrap_type': 'MVS', 'max_leaves': 64}
rfc = CatBoostClassifier(iterations=200,random_state = seed,eval_metric='F1')#CatBoostClassifier(**params_no_its,iterations=4,random_state=seed,)#
rfc.fit(X_train, y_train,eval_set=(X_val,y_val),use_best_model=True)
print(rfc.get_all_params())
from sklearn.metrics import f1_score

plt.bar(X_train.columns,rfc.get_feature_importance())
plt.show()
val_pred=rfc.predict(X_val)
print(rfc.get_feature_importance(),list(X_train.columns))
print(f1_score(y_val,val_pred))

pred = rfc.predict(test)

result=pd.DataFrame({'Event':[],'Q2':[]})
result['Event']=ev
result['Q2']=pred
result.Q2.replace({0:-1}, inplace=True)
result.to_csv('final.csv',index=False)#0.5770318021 nothing 0.5787229071 impulse 0.5809172377 impulse+mass
#TODO copy useless features?
