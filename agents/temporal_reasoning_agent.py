import numpy as np
class TemporalReasoningAgent:
    def reason(self, history):
        x=np.asarray(history); return {'embedding': [float(x.mean()),float(x.std()),float(x[-1])], 'deviation_indicator':float(x[-1]-x.mean())}
