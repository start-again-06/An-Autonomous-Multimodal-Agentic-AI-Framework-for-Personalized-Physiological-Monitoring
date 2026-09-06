import matplotlib.pyplot as plt, pandas as pd, os
def plot_signals(csv):
 os.makedirs("outputs/figures",exist_ok=True); d=pd.read_csv(csv).iloc[:6000]
 fig,ax=plt.subplots(figsize=(10,4)); ax.plot(d.timestamp,d.heart_rate); ax.set(xlabel="Time (s)",ylabel="Heart rate",title="Observed physiological trajectory"); fig.tight_layout(); fig.savefig("outputs/figures/physiological_trajectory.png",dpi=300); plt.close(fig)
