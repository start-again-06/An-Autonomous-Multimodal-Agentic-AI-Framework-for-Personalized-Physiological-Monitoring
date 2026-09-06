class Orchestrator:
    def decide(self,r,quality,dev):
        q=min(v['quality_score'] for v in quality.values())
        if r['total_uncertainty']>.12 or q<.45: return 'VERIFY'
        if r['risk_probability']>=.75: return 'ESCALATE'
        if r['risk_probability']>=.55: return 'ASSESS_RISK'
        if dev>.8: return 'ADJUST_MONITORING'
        return 'WAIT'
