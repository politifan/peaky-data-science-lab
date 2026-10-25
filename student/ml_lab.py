from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier,DummyRegressor
from sklearn.metrics import f1_score,average_precision_score,confusion_matrix,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GroupKFold,cross_validate,GridSearchCV
from sklearn.inspection import permutation_importance
ROOT=Path(__file__).resolve().parent
NUM=['distance','weight','hour'];CAT=['city','weather'];FEATURES=NUM+CAT
def pipeline(model):
    numeric=Pipeline([('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    categorical=Pipeline([('fill',SimpleImputer(strategy='constant',fill_value='missing')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    return Pipeline([('prepare',ColumnTransformer([('numeric',numeric,NUM),('categorical',categorical,CAT)])),('model',model)])
def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);df=pd.read_csv(ROOT/'data/orders.csv').sort_values(['created_at','order_id']).reset_index(drop=True)
    train=df.iloc[:800].copy();valid=df.iloc[800:1000].copy();test=df.iloc[1000:].copy()
    split={'train':train.order_id.tolist(),'valid':valid.order_id.tolist(),'test':test.order_id.tolist()};(out/'split.json').write_text(json.dumps(split),encoding='utf-8')
    models={'baseline':DummyClassifier(strategy='most_frequent'),'linear':LogisticRegression(max_iter=1000,random_state=42),'tree':DecisionTreeClassifier(max_depth=4,min_samples_leaf=15,random_state=42)}
    records=[];fitted={}
    for name,model in models.items():
        m=pipeline(model);cv=cross_validate(m,train[FEATURES],train.late,groups=train.customer_id,cv=GroupKFold(4),scoring=['f1','average_precision'],n_jobs=1)
        m.fit(train[FEATURES],train.late);p=m.predict(valid[FEATURES]);records.append({'name':name,'cv_f1':[float(x) for x in cv['test_f1']],'validation_f1':float(f1_score(valid.late,p,zero_division=0)),'test_used':False,'selected':False});fitted[name]=m
    search=GridSearchCV(pipeline(LogisticRegression(max_iter=1000,random_state=42)),{'model__C':[0.1,1,10]},cv=GroupKFold(4),scoring='f1',n_jobs=1)
    search.fit(train[FEATURES],train.late,groups=train.customer_id)
    best=search.best_estimator_;prob=best.predict_proba(valid[FEATURES])[:,1];thresholds=[]
    for t in [0.2,0.4,0.6,0.8]:
        p=(prob>=t).astype(int);tn,fp,fn,tp=confusion_matrix(valid.late,p,labels=[0,1]).ravel();thresholds.append({'threshold':t,'fp':int(fp),'fn':int(fn),'cost':int(2*fp+20*fn)})
    chosen=min(thresholds,key=lambda r:(r['cost'],-r['threshold']))['threshold']
    # The teaching path freezes this selected linear search before one test call.
    final_prob=best.predict_proba(test[FEATURES])[:,1];final_p=(final_prob>=chosen).astype(int)
    dummy=fitted['baseline'];base=dummy.predict(test[FEATURES]);metrics={'f1':float(f1_score(test.late,final_p,zero_division=0)),'baseline_f1':float(f1_score(test.late,base,zero_division=0)),'average_precision':float(average_precision_score(test.late,final_prob)),'threshold':chosen,'confusion':confusion_matrix(test.late,final_p,labels=[0,1]).tolist()}
    predictions=pd.DataFrame({'order_id':test.order_id,'y':test.late,'probability':final_prob,'prediction':final_p,'city':test.city});predictions.to_csv(out/'predictions.csv',index=False)
    imp=permutation_importance(best,valid[FEATURES],valid.late,scoring='f1',n_repeats=5,random_state=42,n_jobs=1)
    metrics['permutation_importance']={k:float(v) for k,v in zip(FEATURES,imp.importances_mean)}
    reg=pipeline(Ridge(alpha=10));reg.fit(train[FEATURES],train.duration);rp=reg.predict(test[FEATURES]);dr=DummyRegressor(strategy='median');dr.fit(train[NUM].fillna(0),train.duration);bp=dr.predict(test[NUM].fillna(0));metrics['regression']={'mae':float(mean_absolute_error(test.duration,rp)),'rmse':float(np.sqrt(mean_squared_error(test.duration,rp))),'baseline_mae':float(mean_absolute_error(test.duration,bp))}
    records.append({'name':'selected-linear-grid','test_used':True,'selected':True,'best_params':search.best_params_,'thresholds':thresholds})
    joblib.dump(best,out/'model.joblib')
    metadata={'task':'late','seed':42,'features':FEATURES,'data_hash':hashlib.sha256((ROOT/'data/orders.csv').read_bytes()).hexdigest(),'params':search.best_params_,'threshold':chosen,'prediction_mode':'future-period existing-process; GroupKFold as additional CV','versions':{'numpy':np.__version__,'pandas':pd.__version__,'scikit-learn':sklearn.__version__,'joblib':joblib.__version__}}
    for name,value in [('metadata',metadata),('metrics',metrics),('experiments',records)]: (out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    return metrics
if __name__=='__main__':print(run(ROOT/'outputs/ml'))
