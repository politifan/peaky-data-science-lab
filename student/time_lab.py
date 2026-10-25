from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);df=pd.read_csv(ROOT/'data/demand.csv')
    df['lag1']=df.demand.shift(1);df['lag7']=df.demand.shift(7);df['rolling7']=df.demand.shift(1).rolling(7).mean();df=df.dropna().copy()
    train=df.iloc[:-60];test=df.iloc[-60:];cols=['lag1','lag7','rolling7'];m=make_pipeline(StandardScaler(),Ridge(alpha=10));m.fit(train[cols],train.demand);pred=m.predict(test[cols])
    folds=[]
    for tr,va in TimeSeriesSplit(n_splits=3,gap=7).split(train):
        a=train.iloc[tr];b=train.iloc[va];m.fit(a[cols],a.demand);folds.append({'train_max':int(tr.max()),'valid_min':int(va.min()),'mae':float(mean_absolute_error(b.demand,m.predict(b[cols])))})
    result={'n_test':len(test),'baseline_mae':float(mean_absolute_error(test.demand,test.lag7)),'model_mae':float(mean_absolute_error(test.demand,pred)),'folds':folds}
    pd.DataFrame({'date':test.date,'y':test.demand,'prediction':pred,'baseline':test.lag7}).to_csv(out/'predictions.csv',index=False)
    fig,ax=plt.subplots();ax.plot(range(len(test)),test.demand,label='Факт');ax.plot(range(len(test)),pred,label='Прогноз');ax.set(xlabel='День проверочного периода',ylabel='Спрос в заказах');ax.legend();fig.tight_layout();fig.savefig(out/'forecast.png',dpi=160);plt.close(fig)
    (out/'metrics.json').write_text(json.dumps(result),encoding='utf-8');return result
if __name__=='__main__':print(run(ROOT/'outputs/time'))
