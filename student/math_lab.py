"""Linear algebra, gradient and bootstrap with independent numerical checks."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parent
def run():
    X=np.array([[1.,1.],[1.,2.],[1.,3.],[1.,4.]])
    y=np.array([3.,5.,7.,9.]);w=np.array([.4,.7])
    def loss(v):return float(np.mean((X@v-y)**2))
    gradient=2*X.T@(X@w-y)/len(y);eps=1e-6
    numeric=np.array([(loss(w+eps*np.eye(2)[i])-loss(w-eps*np.eye(2)[i]))/(2*eps) for i in range(2)])
    np.testing.assert_allclose(gradient,numeric,rtol=1e-7)
    next_w=w-.01*gradient;assert loss(next_w)<loss(w)
    optimum=np.linalg.lstsq(X,y,rcond=None)[0];np.testing.assert_allclose(optimum,[1.,2.])
    rng=np.random.default_rng(4217);sample=np.array([1.,2.,3.,4.,12.])
    draws=rng.choice(sample,size=(2000,len(sample)),replace=True).mean(axis=1)
    low,high=np.quantile(draws,[.025,.975]);assert low<sample.mean()<high
    cov=np.cov(X[:,1],y,ddof=1)[0,1];corr=np.corrcoef(X[:,1],y)[0,1]
    return {'gradient':gradient.tolist(),'finite_difference':numeric.tolist(),'mse_before':loss(w),
            'mse_after_one_step':loss(next_w),'least_squares':optimum.tolist(),'covariance':float(cov),
            'correlation':float(corr),'bootstrap_percentile_95':[float(low),float(high)],
            'bootstrap_seed':4217,'resamples':2000,'limitations':'Independent illustrative observations; bootstrap does not repair biased sampling; correlation is not causality.'}
if __name__=='__main__':
    out=ROOT/'outputs';out.mkdir(exist_ok=True);r=run();(out/'math.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(r)
