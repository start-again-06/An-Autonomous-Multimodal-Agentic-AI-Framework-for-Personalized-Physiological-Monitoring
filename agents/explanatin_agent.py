class ExplanationAgent:
    def explain(self,r,quality,dev):
        top=max(quality,key=lambda k:quality[k]['quality_score'])
        return {'risk_level':r['risk_level'],'risk_probability':r['risk_probability'],'confidence':r['confidence'],'top_contributing_modalities':[top],'baseline_deviation':dev,'uncertainty':r['total_uncertainty'],'text':f"{r['risk_level']} was associated with a baseline deviation of {dev:.2f}; the most reliable modality was {top}. This is a simulated research event, not a clinical diagnosis."}
