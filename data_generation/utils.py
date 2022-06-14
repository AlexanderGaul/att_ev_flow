import numpy as np
from skimage import measure

def get_connected_components(mask):
    comps = measure.label(mask)
    comp_ids = np.unique(comps)
    background = np.unique(comps[~mask])
    assert background[0] == 0 and len(background) == 1
    comp_ids = np.array([id for id in comp_ids if id not in background])
    return comps, comp_ids