from pathlib import Path
import json,numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parent
def run(seed=42,n=80,effect=1.5):
    rng=np.random.default_rng(seed);a=rng.normal(30,7,n);b=rng.normal(30+effect,7,n)
    test=stats.ttest_ind(a,b,equal_var=False)
    se=float(a.std(ddof=1)/np.sqrt(n));ci=stats.t.interval(0.95,n-1,loc=a.mean(),scale=se)
    return {'seed':seed,'n':n,'mean_a':float(a.mean()),'median_a':float(np.median(a)),'variance_a':float(a.var(ddof=1)),'se_a':se,'ci_a':[float(x) for x in ci],'effect':float(b.mean()-a.mean()),'welch_p':float(test.pvalue),'assumptions':'independent synthetic groups; no causal claim for observational orders'}
if __name__=='__main__':
    o=ROOT/'outputs';o.mkdir(exist_ok=True);r=run();(o/'statistics.json').write_text(json.dumps(r),encoding='utf-8');print(r)
