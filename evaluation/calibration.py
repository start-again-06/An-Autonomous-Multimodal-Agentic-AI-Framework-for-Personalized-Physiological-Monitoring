import numpy as np
def ece(y,p,bins=10):
 edges=np.linspace(0,1,bins+1); return sum(abs(np.mean(p[m])-np.mean(y[m]))*np.mean(m) for m in [(p>=edges[i])&(p<edges[i+1]) for i in range(bins)] if m.any())
