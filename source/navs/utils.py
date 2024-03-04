from .navs import seqnav as spm
from .navs import perfect_memory as pm
from .navs import infomax

def pick_nav(nav_name=None):
    navs = {'smw': spm, 'asmw': spm.SequentialPerfectMemory, 
            'imax': infomax, 's2s':spm.Seq2SeqPerfectMemory}
    if not navs.get(nav_name):
        raise Exception('Non valid matcher method name')
    return navs.get(nav_name)