import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
     
test_done= pd.read_csv('C:/Kaggle/Walmart_weather/data/test_units_0.csv')
testing = pd.read_csv('C:/Kaggle/Walmart_weather/data/test_to_do.csv')
training = pd.read_csv('C:/Kaggle/Walmart_weather/data/full_train.csv')
columns_list=['ID','STORE_ITEM','DATE_NUMERIC_1_1035', 'DATE_NUMERIC_1_31', 'DAY_NUMERIC_1_7', 'DAY_NUMERIC_1_365', 'MONTH_NUMERIC','HIGH_SNOW','HOT','COLD','UNITS']
training=training[columns_list]
testing=testing[columns_list]
training_units=training[['ID','UNITS']]
def file_prob_0():
    submission = test_done 
    name_file='C:/Kaggle/Walmart_weather/output/predictions_0.csv'
    submission.to_csv(name_file, index=False,header=False)

def make_submission(m, test1, id_, filename):
    prediction =np.expm1(m.predict(test1)) 
    submission = pd.DataFrame({"id":id_, "units":prediction})  
    name_file='C:/Kaggle/Walmart_weather/output/tmp/predictions_'+ filename
 #   submission.to_csv(name_file, index=False,header=False)
    return submission.units

def save_mean_submission(prediction, id_, name_log, filename):
    units =np.mean(prediction,1).astype(int)
    submission = pd.DataFrame({"id":id_, "units":units})  
    name_file='C:/Kaggle/Walmart_weather/output/'+name_log+'_'+filename
    submission.to_csv(name_file, index=False,header=False)

    
def rmsle(y_true, y_pred):
    return np.sqrt(np.sum((np.nan_to_num(np.log1p(y_pred))-np.nan_to_num(np.log1p(y_true)))**2)/np.size(y_true))

store_item_list=list(set(training['STORE_ITEM'].values.tolist()))
#store_item_list=['22_93']
b=[]
random_state = 42
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, random_state=random_state)
######################################################
#params = {'C': 5, 'cache_size': 2048}
#clf = SVR(**params) 
#model='svm'
#name_log='{model}_{C}_n_splits_{n_splits}'.format(model=model, C=params['C'],n_splits=n_splits)
######################################################

#params = {'n_neighbors': 9}
#clf = KNeighborsRegressor(**params) 
#model='knn'
#name_log='{model}_{C}_n_splits_{n_splits}'.format(model=model, C=params['n_neighbors'],n_splits=n_splits)
######################################################
params = {'none':0}
clf = LassoCV(alphas=[5., 1., .5]) 
model='lasso'
name_log='{model}_{C}_n_splits_{n_splits}'.format(model=model, C=params['none'],n_splits=n_splits)
######################################################

fName = open('C:/Kaggle/Walmart_weather/log/'+name_log +'.log', 'w')
df_predict=pd.DataFrame(columns=['ID','UNITS_PRED'])


for cur in store_item_list:
    train=training[training['STORE_ITEM']==cur]
    test=testing[testing['STORE_ITEM']==cur]
    id_=test['ID']
    id_train=train['ID']
    target = train["UNITS"].apply(lambda x: np.log1p(x))
    if test.empty:
        continue
    train=train.drop(['ID','STORE_ITEM', 'UNITS'],1).values
    test=test.drop(['ID','STORE_ITEM','UNITS'],1).values
    y=target.values
    id_=id_.values
    id_train=id_train.values

    scaler = StandardScaler()
    train = scaler.fit_transform(train)     
    test = scaler.transform(test)
       
    ind = 0
    a = []
    predict=np.zeros(shape=(np.shape(test)[0], n_splits))
    predict_train=np.zeros(shape=(np.shape(train)[0], n_splits))
    for i, (train_index, test_index) in enumerate(skf.split(train, y)):
#        print ('training set = ', str(ind))
        fit = clf.fit(train[train_index], y[train_index])
        predict[:,i]=make_submission(clf, test,id_, name_log + '_{ind}.csv'.format(ind=ind))
        predict_train[:,i]=np.expm1(clf.predict(train))
        score = rmsle(y[test_index], fit.predict(train[test_index]))
        a+=[score]
        b += [score]
        ind += 1
    cur_submission=pd.DataFrame({'ID':id_train, 'UNITS_PRED':np.mean(predict_train,1).astype(int)})     
    save_mean_submission(predict, id_, name_log, cur+'.csv')  

    df_predict=pd.concat([df_predict,cur_submission])

    log_text = 'Score {} size {} store_item {} \n'.format(np.mean(a),np.size(y), cur)
    fName.write(log_text)
    print ('Score {} size {} id {}'.format(np.mean(b), np.size(y), cur))   
  
check_score= df_predict.merge(training_units,  on='ID')[['ID', 'UNITS','UNITS_PRED']]  
log_text='RMSLE on the training set is: '+ str(rmsle(check_score.UNITS, check_score.UNITS_PRED))
fName.write(log_text)
print(log_text)
log_text='\nMean RMSLE on the training set is: '+str(np.mean(b)) + ' std = ' + str(np.std(b))
fName.write(log_text)
fName.close()
file_prob_0()

