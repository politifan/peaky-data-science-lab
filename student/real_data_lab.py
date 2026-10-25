"""Two real historical datasets, offline CSV, train-only preprocessing and CV."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,confusion_matrix
import joblib

ROOT=Path(__file__).resolve().parent
def run(out=None):
    out=Path(out or ROOT/'outputs/real-data');out.mkdir(parents=True,exist_ok=True)
    results={}
    for name in ['iris','wine']:
        path=ROOT/'data'/('real-'+name+'.csv');frame=pd.read_csv(path)
        X=frame.drop(columns=['row_id','target']);y=frame.target
        train,test=train_test_split(np.arange(len(frame)),test_size=.25,random_state=4217,stratify=y)
        cv=StratifiedKFold(4,shuffle=True,random_state=4217)
        # Hyperparameters and model family are selected only inside train.
        candidates=[('constant',DummyClassifier(strategy='most_frequent'),{}),
                    ('linear',LogisticRegression(max_iter=2000),{'logisticregression__C':[.1,1,10]}),
                    ('tree',DecisionTreeClassifier(random_state=4217),{'decisiontreeclassifier__max_depth':[2,4,6]})]
        experiments=[];models={}
        for key,model,grid in candidates:
            pipe=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),model)
            search=GridSearchCV(pipe,grid,cv=cv,scoring='f1_macro',refit=True)
            search.fit(X.iloc[train],y.iloc[train]);models[key]=search.best_estimator_
            experiments.append({'model':key,'cv_macro_f1':float(search.best_score_),'params':search.best_params_})
        selected=max(experiments,key=lambda r:r['cv_macro_f1'])['model']
        model=models[selected];pred=model.predict(X.iloc[test])
        dest=out/name;dest.mkdir(exist_ok=True);joblib.dump(model,dest/'model.joblib')
        np.testing.assert_array_equal(joblib.load(dest/'model.joblib').predict(X.iloc[test]),pred)
        pd.DataFrame({'row_id':frame.iloc[test].row_id,'target':y.iloc[test],'prediction':pred}).to_csv(dest/'predictions.csv',index=False)
        split={'train_ids':frame.iloc[train].row_id.tolist(),'test_ids':frame.iloc[test].row_id.tolist()}
        (dest/'split.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
        results[name]={'rows':len(frame),'features':len(X.columns),'missing':int(X.isna().sum().sum()),
                       'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'experiments':experiments,
                       'selected':selected,'test_macro_f1':float(f1_score(y.iloc[test],pred,average='macro')),
                       'confusion':confusion_matrix(y.iloc[test],pred).tolist(),'reload_equal':True,
                       'test_used_for_selection':False,'seed':4217}
        # Test is evaluated once after selection, not repeatedly to choose a winner.
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    return results
if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
