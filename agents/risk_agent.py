from models.uncertainty import hybrid_uncertainty
class RiskAssessmentAgent:
    def assess(self,p):
        u=hybrid_uncertainty(p); r=u['predictive_mean']; level='Normal' if r<.3 else 'Mild Deviation' if r<.55 else 'Elevated Risk' if r<.75 else 'High Risk'; return {'risk_probability':r,'predicted_class':int(r>=.5),'risk_level':level,**u}
