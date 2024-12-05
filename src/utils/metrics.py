import numpy as np

def KLD(p,q):

    slf_ent = -1 * (q * np.log(q) + (1 - q) * np.log(1 - q))
    crs_ent = -1 * (q * np.log(p) + (1 - q) * np.log(1 - p))

    return crs_ent - slf_ent