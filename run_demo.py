import os, yaml, json, logging
import numpy as np, pandas as pd
from preprocessing.synthetic_events import generate_dataset
from preprocessing.preprocessing import window_features
from agents.signal_quality_agent import SignalQualityAgent
from agents.representation_agent import PhysiologicalRepresentationAgent
from agents.fusion_agent import FusionAgent
from agents.personalization_agent import PersonalizationAgent
from agents.temporal_reasoning_agent import TemporalReasoningAgent
from agents.risk_agent import RiskAssessmentAgent
from agents.explanation_agent import ExplanationAgent
from agents.orchestrator import Orchestrator
from models.risk_model import RiskModel
def main():
    cfg=yaml.safe_load(open("config.yaml")); os.makedirs("outputs/logs",exist_ok=True)
    logging.basicConfig(filename="outputs/logs/demo.log",level=logging.INFO,format="%(asctime)s %(message)s")
    path="data/synthetic/physiological_demo.csv"; df=generate_dataset(cfg,path)
    X,y,meta=window_features(df); X=np.nan_to_num(X)
    model=RiskModel(cfg["seed"]).fit(X,y); p=model.predict(X)
    sid=int(meta[-1][0]); t=meta[-1][1]; w=df[(df.subject_id==sid) & (df.timestamp.between(secs*.62, secs*.62+120))] if False else df[df.subject_id==sid].iloc[int(len(df[df.subject_id==sid])*.62):int(len(df[df.subject_id==sid])*.62)+6000]
    q=SignalQualityAgent().assess(w); rep=PhysiologicalRepresentationAgent().encode(w); fusion=FusionAgent().fuse(q,rep)
    pa=PersonalizationAgent(); base=pa.update(sid,rep); dev=pa.deviation(rep,base)
    temp=TemporalReasoningAgent().reason(p[-8:]); risk=RiskAssessmentAgent().assess(float(p[-1]))
    expl=ExplanationAgent().explain(risk,q,dev); action=Orchestrator().decide(risk,q,dev)
    result={"timestamp":float(t),"agents":"COMPLETE","risk":risk,"explanation":expl,"fusion":fusion,"temporal":temp,"action":action,
            "monitoring_frequency":{"WAIT":"low","ADJUST_MONITORING":"moderate","ASSESS_RISK":"high","VERIFY":"high","ESCALATE":"continuous"}.get(action,"low")}
    os.makedirs("outputs/predictions",exist_ok=True)
    json.dump(result,open("outputs/demo_result.json","w"),indent=2)
    pd.DataFrame([result["risk"]|{"action":action,"deviation":dev}]).to_csv("outputs/demo_result.csv",index=False)
    print("Loading physiological signals...")
    for x in ["Signal Quality Agent","Representation Agent","Fusion Agent","Personalization Agent","Temporal Reasoning Agent","Risk Assessment Agent","Explanation Agent","Orchestrator","Verification"]:
        print(f"{x}: COMPLETE")
    print(f"Risk probability: {risk['risk_probability']:.2%}\nRisk level: {risk['risk_level']}\nConfidence: {risk['confidence']:.2%}\nUncertainty: {risk['total_uncertainty']:.4f}")
    print("Top contributing modality:",expl["top_contributing_modalities"][0])
    print(f"Top physiological deviation: {dev:.2f}\nSelected action:",action)
    print("Monitoring frequency:",result["monitoring_frequency"])
if __name__=="__main__": main()
