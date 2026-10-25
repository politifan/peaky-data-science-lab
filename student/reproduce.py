from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd,joblib
ROOT=Path(__file__).resolve().parent
def reproduce(directory):
    d=Path(directory);meta=json.loads((d/'metadata.json').read_text(encoding='utf-8'));split=json.loads((d/'split.json').read_text(encoding='utf-8'))
    source=ROOT/'data/orders.csv'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=meta['data_hash']:raise AssertionError('Original input data changed')
    if meta.get('task','late')!='late':raise ValueError('This replay supports classification late only')
    train,valid,test=[list(map(str,split[k])) for k in ['train','valid','test']]
    if not test:raise AssertionError('Empty test split cannot confirm replay')
    if any(len(ids)!=len(set(ids)) for ids in [train,valid,test]):raise AssertionError('Repeated split key')
    if set(train)&set(valid) or set(train)&set(test) or set(valid)&set(test):raise AssertionError('Split overlap')
    full=pd.read_csv(source,dtype={'order_id':'string'})
    if full.order_id.isna().any() or full.order_id.duplicated().any():raise AssertionError('Input order_id must be unique')
    df=full.set_index('order_id').loc[test]
    saved=pd.read_csv(d/'predictions.csv',dtype={'order_id':'string'})
    if saved.order_id.tolist()!=test:raise AssertionError('Prediction keys/order differ from test split')
    # Load only the model produced by your own training run.
    m=joblib.load(d/'model.joblib');prob=m.predict_proba(df[meta['features']])[:,1]
    old=saved.probability.to_numpy(dtype=float)
    if len(prob)!=len(old) or not np.isfinite(prob).all() or not np.isfinite(old).all() or not np.allclose(prob,old,rtol=0,atol=1e-12):raise AssertionError('Restart probabilities changed or missing')
    prediction=(prob>=meta['threshold']).astype(int)
    if not np.array_equal(saved.prediction.to_numpy(),prediction):raise AssertionError('Saved class differs from frozen threshold')
    if not np.array_equal(saved.y.to_numpy(),df.late.to_numpy()):raise AssertionError('Saved target differs from source')
    return {'restart_match':True,'n':len(prob)}
if __name__=='__main__':print(reproduce(sys.argv[1] if len(sys.argv)>1 else ROOT/'outputs/ml'))
