def test_imports():
 from agents.orchestrator import Orchestrator
 assert Orchestrator().decide({'risk_probability':.1,'total_uncertainty':.01},{'ecg':{'quality_score':1},'ppg':{'quality_score':1},'acc':{'quality_score':1}},0)=='WAIT'
