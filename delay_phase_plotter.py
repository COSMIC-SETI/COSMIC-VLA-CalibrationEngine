import sys
import os
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from calibrate_uvh5 import calibrate_uvh5
import calib_util

def delay_optimize(data):
    # Takes 1-D complex values, finds optimized delay in units of inv bw
    def fn(t,d):
        n = len(d)
        x = d*np.exp(2.0j*np.pi*t*np.arange(n)/n)
        return -1.0*np.abs(x.sum())
    fdat = np.abs(np.fft.ifft(data))
    t0 = float(fdat.argmax())
    if t0 > len(fdat)/2.0:
        t0 -= len(fdat)
    t_opt = scipy.optimize.fmin(fn,t0,args=(data,),ftol=1e-7,disp=False)
    return t_opt

#def calc_delays(visdata, ant_indices, refant=10):
#    # ant_indices is array of antenna pairs
#    # Assume axes (bl,time,freq,pol)
#    nbl = visdata.shape[0] # should equal len(ant_indices)
#    for ibl in range(nbl):

# Load file
fname = sys.argv[1]
c = calibrate_uvh5(fname,None)
nant = len(c.metadata['ant_curr'])

# test
iref = 10
for iant in range(nant):
    if iant<iref:
        ibl = c.ant_indices.index([iant,iref])
    elif iant>iref:
        ibl = c.ant_indices.index([iref,iant])
    else:
        pass
    avgdata = c.vis_data.mean(1) # avg time
    d0 = delay_optimize(avgdata[ibl,:,0])
    d1 = delay_optimize(avgdata[ibl,:,-1])
    print(c.metadata['ant_curr'][iant],d0,d1)
