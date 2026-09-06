import numpy as np
def hybrid_uncertainty(p, rng=None, draws=20):
    rng=np.random.default_rng(42) if rng is None else rng
    sims=np.clip(p+rng.normal(0,.03,draws),0,1)
    epi=float(np.var(sims)); ale=float(p*(1-p)); total=float(epi+ale)
    return {"predictive_mean":float(np.mean(sims)),"predictive_variance":float(np.var(sims)),
            "epistemic_uncertainty":epi,"aleatoric_uncertainty":ale,"total_uncertainty":total,
            "confidence":float(1-min(1,total))}
