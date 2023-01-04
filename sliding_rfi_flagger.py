import numpy as np
from mad import median_absolute_deviation as mad

def flag_rfi(data, winSize, clip=3):
          
    spec = np.abs(data)
    smth = spec*0.0

    # Compute the smoothed bandpass model
    for i in range(smth.size):
        mn = int(max([0, i-winSize/2]))
        mx = int(min([i+winSize/2+1, smth.size]))
        smth[i] = np.median(spec[mn:mx])      
           

    diff = (spec-smth) 
    med = np.median(diff)
    #sig_mn = np.std(diff)
    sig_md = mad(diff)
          
    bad = np.argwhere(abs(diff-med) > clip*sig_md)

    return  bad


   

