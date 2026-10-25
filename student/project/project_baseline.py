from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parent
def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);df=pd.read_csv(ROOT.parent/'data/orders.csv').sort_values(['created_at','order_id']);train=df.iloc[:800];test=df.iloc[1000:]
    model=DummyClassifier(strategy='most_frequent');model.fit(train[['hour']],train.late);p=model.predict(test[['hour']]);r={'baseline_f1':float(f1_score(test.late,p,zero_division=0)),'n_train':len(train),'n_test':len(test)};(out/'baseline.json').write_text(json.dumps(r),encoding='utf-8');return r
if __name__=='__main__':print(run(ROOT/'outputs'))
