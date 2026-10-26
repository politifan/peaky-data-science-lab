"""A frozen learner plan, validation selection, then one final holdout.

Run: python own_experiment.py project/my-plan.json outputs/my-attempt
Only load joblib files created by your own training.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np,pandas as pd,joblib,sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,f1_score,average_precision_score
ROOT=Path(__file__).resolve().parent
NUM=['distance','weight','hour'];CAT=['city','weather'];ALLOWED=NUM+CAT
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def validate(plan):
    features=plan['features'];candidates=plan['candidates'];thresholds=plan['thresholds']
    if not features or len(set(features))!=len(features) or set(features)-set(ALLOWED):raise ValueError('Features must be a nonempty unique subset of decision-time inputs')
    if len(plan.get('hypothesis','').strip())<20:raise ValueError('Write your own question before the run')
    if plan.get('protocol')!='future-existing-process':raise ValueError('This path evaluates a future period of the same synthetic process')
    if len(candidates)<2 or not any(c['kind']=='baseline' for c in candidates) or all(c['kind']=='baseline' for c in candidates):raise ValueError('Compare a baseline and a substantive candidate')
    if len({c['name'] for c in candidates})!=len(candidates):raise ValueError('Unique candidate names required')
    if not thresholds or any(type(t) not in [int,float] or not 0<t<1 for t in thresholds):raise ValueError('Thresholds must be frozen numbers between 0 and 1')
    availability=plan['availability']
    if set(availability)!=set(features) or any(v!='at_decision' for v in availability.values()):raise ValueError('Declare availability of every feature at decision time')
    for c in candidates:
        if c['kind'] not in ['baseline','linear','tree']:raise ValueError('Unsupported candidate family')
        if c['kind']=='linear' and (type(c.get('C')) not in [int,float] or c['C']<=0):raise ValueError('C must be positive')
        if c['kind']=='tree' and (type(c.get('max_depth')) is not int or not 1<=c['max_depth']<=20):raise ValueError('max_depth must be 1..20')
    return features
def make_model(candidate,features):
    numeric=[c for c in features if c in NUM];categorical=[c for c in features if c in CAT];parts=[]
    if numeric:parts.append(('numeric',Pipeline([('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric))
    if categorical:parts.append(('categorical',Pipeline([('fill',SimpleImputer(strategy='constant',fill_value='missing')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical))
    kind=candidate['kind']
    model=DummyClassifier(strategy='prior') if kind=='baseline' else LogisticRegression(C=candidate['C'],max_iter=1000,random_state=42) if kind=='linear' else DecisionTreeClassifier(max_depth=candidate['max_depth'],min_samples_leaf=15,random_state=42)
    return Pipeline([('prepare',ColumnTransformer(parts)),('model',model)])
def cost(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return dict(cost=int(2*fp+20*fn),fp=int(fp),fn=int(fn))
def run(plan_path,out):
    plan=read(plan_path);features=validate(plan);out=Path(out)
    if out.exists():raise FileExistsError('Use a new attempt directory; previous results are preserved')
    source=ROOT/'data/orders.csv';df=pd.read_csv(source).sort_values(['created_at','order_id']).reset_index(drop=True)
    if len(df)<1100 or df.order_id.isna().any() or df.order_id.duplicated().any():raise ValueError('Unexpected source keys/size')
    train,valid,test=df.iloc[:800],df.iloc[800:1000],df.iloc[1000:]
    if not train.created_at.max()<valid.created_at.min() or not valid.created_at.max()<test.created_at.min():raise ValueError('Time boundary is not strictly separated')
    out.mkdir(parents=True);dump(out/'plan.json',plan)
    plan_hash=hashlib.sha256((out/'plan.json').read_bytes()).hexdigest()
    split={name:part.order_id.tolist() for name,part in [('train',train),('valid',valid),('test',test)]};dump(out/'split.json',split)
    rows=[];fitted={}
    for candidate in plan['candidates']:
        m=make_model(candidate,features);m.fit(train[features],train.late);prob=m.predict_proba(valid[features])[:,1];fitted[candidate['name']]=m
        for threshold in plan['thresholds']:
            rows.append(dict(name=candidate['name'],threshold=threshold,**cost(valid.late,(prob>=threshold).astype(int)),test_used=False))
    chosen=min(rows,key=lambda r:(r['cost'],r['name'],-r['threshold']));dump(out/'validation.json',dict(rows=rows,selected=chosen,rule='min cost, then name, then higher threshold'))
    # The decision is saved before touching the holdout targets or predictions.
    best=fitted[chosen['name']];prob=best.predict_proba(test[features])[:,1];pred=(prob>=chosen['threshold']).astype(int)
    baseline_name=next(c['name'] for c in plan['candidates'] if c['kind']=='baseline');baseprob=fitted[baseline_name].predict_proba(test[features])[:,1]
    base_choice=min([r for r in rows if r['name']==baseline_name],key=lambda r:(r['cost'],-r['threshold']));base=(baseprob>=base_choice['threshold']).astype(int)
    metrics=dict(**cost(test.late,pred),f1=float(f1_score(test.late,pred,zero_division=0)),average_precision=float(average_precision_score(test.late,prob)),baseline_cost=cost(test.late,base)['cost'],n_test=len(test))
    pd.DataFrame(dict(order_id=test.order_id,y=test.late,probability=prob,prediction=pred)).to_csv(out/'predictions.csv',index=False);joblib.dump(best,out/'model.joblib')
    dump(out/'metadata.json',dict(task='late',features=features,data_hash=hashlib.sha256(source.read_bytes()).hexdigest(),plan_hash=plan_hash,threshold=chosen['threshold'],selected=chosen['name'],prediction_mode=plan['protocol'],seed=42,versions={'numpy':np.__version__,'pandas':pd.__version__,'scikit-learn':sklearn.__version__,'joblib':joblib.__version__}))
    dump(out/'metrics.json',metrics);return metrics
if __name__=='__main__':print(run(sys.argv[1],sys.argv[2]))
