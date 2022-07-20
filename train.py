# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import math,random,os

pd.set_option('display.max_columns', None)

seed=42
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(seed)

"""Здесь мы открываем файлы из папки data: данные для обучения модели и для итогового результата."""

train = pd.read_excel('For_model_labled.xlsx')
test = pd.read_excel('For_check_unlabled.xlsx')



train=train.drop_duplicates()

train["M"]=train["M"].fillna(train["M"].median())
test["M"]=test["M"].fillna(test["M"].median())

y = train['Q2']
X = train.drop(columns = ['Q2'])

X=X.drop('Event',axis=1)
ev=test['Event']
test=test.drop('Event',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed, test_size = 0.2)

"""Модель, выбранная в бейзлайне - случайный лес. Она не является оптимальной, но быстро обучается и позволяет получить результат, который выше случайного. Можете начать улучшать бейзлайлн с того, что попробовать изменить гиперпараметры представленного случайного леса."""

from catboost import CatBoostClassifier

rfc = CatBoostClassifier(iterations=200,random_state = seed,eval_metric='F1')
rfc.fit(X_train, y_train,eval_set=(X_test,y_test),use_best_model=True)

pred = rfc.predict(test)

result=pd.DataFrame({'Event':[],'Q2':[]})
result['Event']=ev
result['Q2']=pred
result.to_csv('final.csv',index=False)
