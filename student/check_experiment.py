"""Check protocol artifacts and replay; declarations are not real-world provenance."""
from own_experiment import read,validate,ROOT
from reproduce import reproduce
from pathlib import Path
import hashlib,sys,pandas as pd
def check(directory):
    d=Path(directory);plan=read(d/'plan.json');features=validate(plan);meta=read(d/'metadata.json');split=read(d/'split.json');selection=read(d/'validation.json')
    if hashlib.sha256((d/'plan.json').read_bytes()).hexdigest()!=meta['plan_hash']:raise AssertionError('Plan changed after training')
    if features!=meta['features']:raise AssertionError('Model features differ from frozen plan')
    df=pd.read_csv(ROOT/'data/orders.csv',dtype={'order_id':'string'}).sort_values(['created_at','order_id'])
    for name,part in [('train',df.iloc[:800]),('valid',df.iloc[800:1000]),('test',df.iloc[1000:])]:
        if list(map(str,split[name]))!=part.order_id.tolist():raise AssertionError('Frozen time split changed: '+name)
    rows=selection['rows'];best=min(rows,key=lambda r:(r['cost'],r['name'],-r['threshold']))
    if any(r['test_used'] for r in rows) or best!=selection['selected'] or best['name']!=meta['selected'] or best['threshold']!=meta['threshold']:raise AssertionError('Selection contradicts saved validation protocol')
    result=reproduce(d)
    return dict(**result,protocol_artifacts=True,availability='synthetic declared contract; real-source timing needs evidence')
if __name__=='__main__':print(check(sys.argv[1]))
