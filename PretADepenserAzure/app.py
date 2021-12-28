#commands to execute
#py -3 -m venv .venv && .venv\scripts\activate



from flask import Flask, flash, redirect, render_template, request, session, abort, url_for, request
#!/usr/bin/env python
# coding: utf-8

# usual data science stack in python
import numpy as np
import pandas as pd

#import os
# imports of need modules in sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
#import lightgbm as lgb
#import xgboost as xgb
app = Flask(__name__)

#path_train = os.path.join('./data/', 'application_train.csv')
#path_test = os.path.join( './data/', 'application_test.csv')

def preparationData():
     # Any results you write to the current directory are saved as output.
    # load main datasets
    '''
    apptrainPart1_1 , apptrainPart1_2, apptrainPart1_3, apptrainPart1_4=pd.read_csv("./data/application_trainPart1_1.csv"),pd.read_csv("./data/application_trainPart1_2.csv"),pd.read_csv("./data/application_trainPart1_3.csv"),pd.read_csv("./data/application_trainPart1_4.csv")
    apptrainPart2_1 , apptrainPart2_2, apptrainPart2_3, apptrainPart2_4=pd.read_csv("./data/application_trainPart2_1.csv"),pd.read_csv("./data/application_trainPart2_2.csv"),pd.read_csv("./data/application_trainPart2_3.csv"),pd.read_csv("./data/application_trainPart2_4.csv")
    apTrainReunated1_1=pd.concat([apptrainPart1_1,apptrainPart1_2])
    apTrainReunated1_2=pd.concat([apptrainPart1_3,apptrainPart1_4])
    apTrainReunated2_1=pd.concat([apptrainPart2_1,apptrainPart2_2])
    apTrainReunated2_2=pd.concat([apptrainPart2_3,apptrainPart2_4])
    apTrainReunated1=pd.concat([apTrainReunated1_1,apTrainReunated1_2])
    apTrainReunated2=pd.concat([apTrainReunated2_1,apTrainReunated2_2])
    app_train=pd.concat([apTrainReunated1,apTrainReunated2])
    aptestPart1, aptestPart2=pd.read_csv("./data/application_testPart1.csv"), pd.read_csv("./data/application_testPart2.csv")
    app_test=pd.concat([aptestPart1,aptestPart2])
    '''
    app_train=pd.read_csv("./data/application_train.csv")
    app_test=pd.read_csv("./data/application_test.csv")
    app_train=app_train[["SK_ID_CURR","TARGET","DAYS_BIRTH", "DAYS_ID_PUBLISH","FLAG_EMAIL","OCCUPATION_TYPE","AMT_INCOME_TOTAL","AMT_GOODS_PRICE","AMT_ANNUITY","FLAG_OWN_CAR","AMT_CREDIT","HOUR_APPR_PROCESS_START","CODE_GENDER","NAME_CONTRACT_TYPE","CNT_CHILDREN","DAYS_EMPLOYED"]]
    app_test=app_test[["SK_ID_CURR","FLAG_EMAIL","DAYS_BIRTH", "DAYS_ID_PUBLISH","OCCUPATION_TYPE","AMT_INCOME_TOTAL","AMT_GOODS_PRICE","AMT_ANNUITY","FLAG_OWN_CAR","AMT_CREDIT","HOUR_APPR_PROCESS_START","CODE_GENDER","NAME_CONTRACT_TYPE","CNT_CHILDREN","DAYS_EMPLOYED"]]


    # cols_to_drop = list((app_train.isnull().sum() > 75000).index)
    cols_to_drop = [c for c in app_train.columns if app_train[c].isnull().sum() > 75000]
    app_train, app_test = app_train.drop(cols_to_drop, axis=1), app_test.drop(cols_to_drop, axis=1)
    obj_cols = app_train.select_dtypes('object').columns
    # filling string cols with 'Not specified' 
    app_train[obj_cols] = app_train[obj_cols].fillna('Not specified')
    app_test[obj_cols] = app_test[obj_cols].fillna('Not specified')
    float_cols = app_train.select_dtypes('float').columns
    # filling float values with median of train (not test)
    app_train[float_cols] = app_train[float_cols].fillna(app_train[float_cols].median())
    app_test[float_cols] = app_test[float_cols].fillna(app_test[float_cols].median())
    # Create an anomalous flag column
    app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
    # Replace the anomalous values with nan
    app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
    app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
    app_train[float_cols] = app_train[float_cols].apply(pd.to_numeric, errors='coerce')
    app_train = app_train.fillna(app_train.median())
    app_test[float_cols] = app_test[float_cols].apply(pd.to_numeric, errors='coerce')
    app_test = app_train.fillna(app_test.median())
    app_train[float_cols] = app_train[float_cols].apply(pd.to_numeric, errors='coerce')
    app_train = app_train.fillna(app_train.median())

    app_test[float_cols] = app_test[float_cols].apply(pd.to_numeric, errors='coerce')
    app_test = app_train.fillna(app_test.median())
    app_train = pd.get_dummies(data=app_train, columns=obj_cols)
    app_test = pd.get_dummies(data=app_test, columns=obj_cols)
    return [app_train,app_test,float_cols]






def getNewUser(res):
    var_SK_ID_CURR=res[0]['SK_ID_CURR'].tail(1).item()+1
    var_AMT_GOODS_PRICE=res[0]['AMT_GOODS_PRICE'].mean()
    var_AMT_ANNUITY=res[0]['AMT_ANNUITY'].mean()
    var_AMT_CREDIT=res[0]['AMT_CREDIT'].mean()
    var_HOUR_APPR_PROCESS_START=res[0]['HOUR_APPR_PROCESS_START'].mean()
    var_DAYS_EMPLOYED=res[0]['DAYS_EMPLOYED'].mean()
    var_DAYS_EMPLOYED_ANOM=res[0]['DAYS_EMPLOYED_ANOM'].mean()

    daybirth=input("type you birthday(ex:15/12/1990): ")
    from datetime import date
    strinDate=daybirth
    day_month_year= strinDate.split("/")
    d0 = date(np.int(day_month_year[2]),np.int(day_month_year[1]),np.int(day_month_year[0]))
    d1 = date.today()
    delta = d0 - d1
    print(delta.days)
    var_DAYS_BIRTH=delta.days
    var_DAYS_ID_PUBLISH=0

    myemail=input("type you email: ")
    if (myemail ==""):
        var_FLAG_EMAIL=0
    else:
        var_FLAG_EMAIL=1
    myIncome=input("type you income: ")
    var_AMT_INCOME_TOTAL=np.int(myIncome)
    myChilds=input("Do you have children? how many?: ")
    var_CNT_CHILDREN=np.int(myChilds)
    myCar=input("Do you have a car?: ")
    if myCar=="yes":
        var_FLAG_OWN_CAR_N=0
        var_FLAG_OWN_CAR_Y=1
    else:
        var_FLAG_OWN_CAR_N=1
        var_FLAG_OWN_CAR_Y=0
    mySexe=input("Are you a man or a woman?: ")
    if mySexe=="man":
        var_CODE_GENDER_F=0
        var_CODE_GENDER_M=1
        var_CODE_GENDER_XNA=0
    else:
        var_CODE_GENDER_F=1
        var_CODE_GENDER_M=0
        var_CODE_GENDER_XNA=0
    myTypeOfloans=input("what type of loans(cash or revolving)?: ")
    if myTypeOfloans=="cash":
        var_NAME_CONTRACT_TYPE_Cash_loans=1
        var_NAME_CONTRACT_TYPE_Revolving_loans=0
    else:
        var_NAME_CONTRACT_TYPE_Cash_loans=0
        var_NAME_CONTRACT_TYPE_Revolving_loans=1
    dfNewUser=pd.DataFrame(data=[[var_SK_ID_CURR,var_DAYS_BIRTH,var_DAYS_ID_PUBLISH,var_FLAG_EMAIL,var_AMT_INCOME_TOTAL,var_AMT_GOODS_PRICE,var_AMT_ANNUITY,var_AMT_CREDIT,var_HOUR_APPR_PROCESS_START,var_CNT_CHILDREN,var_DAYS_EMPLOYED,var_DAYS_EMPLOYED_ANOM,var_FLAG_OWN_CAR_N,var_FLAG_OWN_CAR_Y,var_CODE_GENDER_F,var_CODE_GENDER_M,var_CODE_GENDER_XNA,var_NAME_CONTRACT_TYPE_Cash_loans,var_NAME_CONTRACT_TYPE_Revolving_loans]],columns=["SK_ID_CURR","DAYS_BIRTH","DAYS_ID_PUBLISH","FLAG_EMAIL","AMT_INCOME_TOTAL","AMT_GOODS_PRICE","AMT_ANNUITY","AMT_CREDIT","HOUR_APPR_PROCESS_START","CNT_CHILDREN","DAYS_EMPLOYED","DAYS_EMPLOYED_ANOM","FLAG_OWN_CAR_N","FLAG_OWN_CAR_Y","CODE_GENDER_F","CODE_GENDER_M","CODE_GENDER_XNA","NAME_CONTRACT_TYPE_Cash loans","NAME_CONTRACT_TYPE_Revolving loans"])
    #nl, to scale for better results
    float_cols=res[2]
    feat_to_scale = list(float_cols).copy()
    feat_to_scale.extend(['CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'HOUR_APPR_PROCESS_START'])
    scaler = StandardScaler()
    dfNewUser[feat_to_scale] = scaler.fit_transform(dfNewUser[feat_to_scale])
    return dfNewUser








"""
#################################################################################################################################☺
#################################################################################################################################☺
#################################################################################################################################☺
#################################################################################################################################☺
"""


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/dashboard/<string:name>/")
def dashboard(name):
    return render_template('index.html',name=name)

@app.route('/API/<idUser>')
def MainAppData(idUser):
    
    print("the idUser is:",type(int(idUser)))
    print("the idUser should be:",type(100002))
    res=preparationData()
    app_train=res[0]
    app_test=res[1]
    float_cols=res[2]
    # back up of the target /  need to keep this information
    y = app_train.TARGET
    app_train = app_train.drop(columns=['TARGET'])
    app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)
    feat_to_scale = list(float_cols).copy()
    feat_to_scale.extend(['CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'HOUR_APPR_PROCESS_START'])
    scaler = StandardScaler()
    app_train[feat_to_scale] = scaler.fit_transform(app_train[feat_to_scale])
    app_test[feat_to_scale] = scaler.fit_transform(app_test[feat_to_scale])
    currUser=res[0][res[0]['SK_ID_CURR']==int(idUser)].drop(['TARGET'],axis=1)
    if currUser.empty == False:
        #print("the user cur is: ",currUser)
        currUser[feat_to_scale] = scaler.fit_transform(currUser[feat_to_scale])
        X_train, X_test, y_train, y_test = train_test_split(app_train, y)

        ##with random forest
        rlf=RandomForestClassifier(n_estimators=50)
        rlf.fit(X_train[:10000], y_train[:10000])
        finalResProba=rlf.predict_proba(currUser)[:, 1]   

        ##with xgboost mais pas de bon resultats
        #modelxgb = xgb.XGBClassifier(depth=10, iterations= 500, l2_leaf_reg= 9, learning_rate= 0.15)
        #modelxgb.fit(X_train, y_train)
        #finalResProba=modelxgb.predict_proba(currUser)[:, 1]

        ##with lgbm
        #lgbm = lgb.LGBMClassifier(random_state = 50, n_jobs = -1, class_weight = 'balanced')
        #print("jvais bien dans lgbm avant le fit")
        #lgbm.fit(X_train, y_train)
        #finalResProba=lgbm.predict_proba(currUser)[:, 1]
        
        ##with catboost
        #clf = cb.CatBoostClassifier(eval_metric="AUC", depth=10, iterations= 500, l2_leaf_reg= 9, learning_rate= 0.15)
        #clf.fit(X_train, y_train)
        #finalResProba=clf.predict_proba(currUser)[:, 1]
        
        ##with adaboost
        #modelAdaBoost = AdaBoostClassifier(n_estimators=100,learning_rate=0.15)
        #modelAdaBoost.fit(X_train, y_train)
        #finalResProba=modelAdaBoost.predict_proba(currUser)[:, 1]

        ##with gradiant boosting
        #modelGradBoostClf = GradientBoostingClassifier(n_estimators=70, learning_rate=0.15)
        #modelGradBoostClf.fit(X_train, y_train)
        #finalResProba=modelGradBoostClf.predict_proba(currUser)[:, 1]
    else :
        finalResProba=[-1]
        print("jvais bien mais y a pas le user")
    print("finalResProba:",round(finalResProba[0]*100))
    return render_template('myapires.html',results=round(finalResProba[0]*100))
    

@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['name']
      return redirect(url_for('MainAppData',idUser = user))
   else:
      user = request.args.get('name')
      return redirect(url_for('MainAppData',idUser = user))

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)