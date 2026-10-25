from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);df=pd.read_csv(ROOT/'data/orders.csv')
    figures=[]
    fig,ax=plt.subplots();df.isna().sum().plot.bar(ax=ax);ax.set(ylabel='Строки с пропусками',title='Полнота входных данных');fig.tight_layout();fig.savefig(out/'missing.png',dpi=160);plt.close(fig);figures.append('missing.png')
    fig,ax=plt.subplots();ax.hist(df.duration,bins=20);ax.set(xlabel='Время доставки в минутах',ylabel='Заказы');fig.tight_layout();fig.savefig(out/'duration.png',dpi=160);plt.close(fig);figures.append('duration.png')
    fig,ax=plt.subplots();ax.scatter(df.distance,df.duration,alpha=0.25,s=8);ax.set(xlabel='Расстояние в километрах',ylabel='Время доставки в минутах');fig.tight_layout();fig.savefig(out/'distance.png',dpi=160);plt.close(fig);figures.append('distance.png')
    seg=df.groupby('city').late.agg(['size','mean']);seg.to_csv(out/'segments.csv')
    fig,ax=plt.subplots();seg['mean'].plot.bar(ax=ax);ax.set(ylabel='Доля опозданий',xlabel='Город',title='Опоздания и размер каждой группы')
    for i,(_,r) in enumerate(seg.iterrows()):ax.text(i,float(r['mean'])+0.003,'n='+str(int(r['size'])),ha='center',fontsize=9)
    ax.set_ylim(0,max(seg['mean'])*1.15);fig.tight_layout();fig.savefig(out/'cities.png',dpi=160);plt.close(fig);figures.append('cities.png')
    result={'n':len(df),'missing_distance':int(df.distance.isna().sum()),'figures':figures};(out/'eda.json').write_text(json.dumps(result),encoding='utf-8');return result
if __name__=='__main__':print(run(ROOT/'outputs/eda'))
