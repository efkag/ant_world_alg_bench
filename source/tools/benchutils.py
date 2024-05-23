from source.navs import seqnav as spm
from source.navs import perfect_memory as pm
from source.navs import infomax
import json
import os
import pandas as pd

def pick_nav(nav_name=None):
    navs = {'pm':pm.PerfectMemory,
            'smw': spm.SequentialPerfectMemory, 
            'asmw': spm.SequentialPerfectMemory, 
            's2s': spm.Seq2SeqPerfectMemory,
            'imax': infomax.InfomaxNetwork}
    if not navs.get(nav_name):
        raise Exception('Non valid navigator class name')
    return navs.get(nav_name)

def get_logs_dict() -> dict:
    path = os.path.dirname(__file__)
    path = os.path.join(path, 'log_headers.json')
    with open(path, 'r') as f:
        logs = json.load(f)
    return logs

def dict_to_frame(logs: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(logs, orient='index')
    df = df.transpose()
    return df
