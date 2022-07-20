# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import math,random,os
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

c=	299792458
seed=42

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(seed)


train = pd.read_excel('For_model_labled.xlsx')
test = pd.read_excel('For_check_unlabled.xlsx')


train=train.drop_duplicates()


#train["M"]=train["M"].fillna(train["M"].median())
#test["M"]=test["M"].fillna(test["M"].median())

'''train['m1']=(train['E1']/1000)/(931.5**2)
train['m2']=(train['E2']/1000)/(931.5**2)

test['m1']=(test['E1']/1000)/(931.5**2)
test['m2']=(test['E2']/1000)/(931.5**2)

'''
def calc_M(phi1,phi2,eta1,eta2,p1,p2):
    return np.sqrt( 2*p1*p2* ( np.cosh(eta1-eta2) - np.cos( phi1 - phi2 ) ) )


train['M_new']=calc_M(train['phi1'],train['phi2'],train['eta1'],train['eta2'],train['pt1'],train['pt2'])
test['M_new']=calc_M(test['phi1'],test['phi2'],test['eta1'],test['eta2'],test['pt1'],test['pt2'])
train=train.drop('M',axis=1)
test=test.drop('M',axis=1)

train['p1']=(train['px1']**2)+(train['py1']**2)+(train['pz1']**2)
train['p2']=(train['px2']**2)+(train['py2']**2)+(train['pz2']**2)
test['p1']=(test['px1']**2)+(test['py1']**2)+(test['pz1']**2)
test['p2']=(test['px2']**2)+(test['py2']**2)+(test['pz2']**2)

def calc_unstatic_mass(E,p):
    return np.sqrt( E**2 - (p**2 * 931.5**2) )/(c**2)

train['m1_unstatic']=calc_unstatic_mass(train['E1'],train['p1'])
train['m2_unstatic']=calc_unstatic_mass(train['E2'],train['p2'])
test['m1_unstatic']=calc_unstatic_mass(test['E1'],test['p1'])
test['m2_unstatic']=calc_unstatic_mass(test['E2'],test['p2'])
#del train['p1'],train['p2'],test['p1'],test['p2']

unimportant=['px1','px2','py1','py2','pz1','pz2','phi1','phi2']

train=train.drop(columns=unimportant)
test=test.drop(columns=unimportant)





y = train['Q2']
X = train.drop(columns = ['Q2'])

X=X.drop(columns=['Event','Run'])
ev=test['Event']
test=test.drop(columns=['Event','Run'])


X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = seed, test_size = 0.2)

"""Модель, выбранная в бейзлайне - случайный лес. Она не является оптимальной, но быстро обучается и позволяет получить результат, который выше случайного. Можете начать улучшать бейзлайлн с того, что попробовать изменить гиперпараметры представленного случайного леса."""

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
rfc = CatBoostClassifier(iterations=200,random_state = seed,eval_metric='F1')
rfc.fit(X_train, y_train,eval_set=(X_val,y_val),use_best_model=True)
#rfc=RandomForestClassifier(n_estimators=200)
#rfc.fit(X_train,y_train)
from sklearn.metrics import f1_score

plt.bar(X_train.columns,rfc.get_feature_importance())
plt.show()
val_pred=rfc.predict(X_val)
print(f1_score(y_val,val_pred))

pred = rfc.predict(test)

result=pd.DataFrame({'Event':[],'Q2':[]})
result['Event']=ev
result['Q2']=pred
result.to_csv('final.csv',index=False)#0.5770318021 nothing 0.5787229071 impulse 0.5809172377 impulse+mass
