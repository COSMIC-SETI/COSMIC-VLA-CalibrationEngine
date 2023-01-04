"""
Calibration codes written by Paul Demorest
Edited by Savin Shynu Varghese for calibrating the COSMIC data
"""

import numpy as np
from numpy import linalg as linalg_cpu
import cupy as cp
from cupy import linalg as linalg_gpu



# Some routines to derive simple calibration solutions directly
# from data arrays.

def ant2bl(i, j=None):
    """Returns baseline index for given antenna pair.  Will accept
    two args, or a list/tuple/etc.  Uses 0-based indexing"""
    if j is None:
        (a1, a2) = sorted(i[:2])
    else:
        (a1, a2) = sorted((i, j))
    # could raise error if a2==a1, either are negative, etc
    return (a2*(a2-1))//2 + a1


def bl2ant(i):
    """Returns antenna pair for given baseline index.  All are 0-based."""
    a2 = int(0.5*(1.0+np.sqrt(1.0+8.0*i)))
    a1 = i - a2*(a2-1)//2
    return a1, a2



def gaincal_cpu(data, nant, ant_indices, axis=0, ref=0, avg=[], nit=3):
    """Derives amplitude/phase calibration factors from the data array
    for the given baseline axis.  In the returned array, the baseline
    dimension is converted to antenna.  No other axes are modified.
    Note this internally makes a transposed copy of the data so be
    careful with memory usage in the case of large data sets.  A list
    of axes to average over before solving can be given in the avg
    argument (length-1 dimensions are kept so that the solution can be
    applied to the original data)."""
    nbl = data.shape[axis]
    ndim = len(data.shape)
    #(check, nant) = bl2ant(nbl)
    #print(nant)
    #if check != 0:
    #    raise RuntimeError("Specified axis dimension (%d) is not a valid number of baselines" % nbl)
    if avg != []:
        # Average, ignoring zeros
        #norm = np.count_nonzero(data,axis=avg,keepdims=True) # requires numpy 1.19 for keepdims
        norm = np.count_nonzero(data,axis=tuple(avg))
        norm[np.where(norm==0)] = 1
        data = data.sum(axis=tuple(avg), keepdims=True)
        norm = norm.reshape(data.shape) # workaround lack of keepdims
        data = data / norm
    tdata = np.zeros(data.shape[:axis]+data.shape[axis+1:]+(nant, nant),
                     dtype=data.dtype)
    for i in range(nbl):
        #(a0, a1) = bl2ant(i)
        #print(a0, a1)
        [a0, a1] = ant_indices[i]
        if a0 != a1:
            tdata[..., a0, a1] = data.take(i, axis=axis)
            tdata[..., a1, a0] = np.conj(data.take(i, axis=axis))
    for it in range(nit):
        (wtmp, vtmp) = linalg_cpu.eigh(tdata)
        v = vtmp[..., -1].copy()
        w = wtmp[..., -1]
        for i in range(nant):
            tdata[..., i, i] = w*(v.real[..., i]**2 + v.imag[..., i]**2)
    
    #(wtmp, vtmp) = linalg.eigh(tdata)
    #v = vtmp[..., -1].copy()
    #w = wtmp[..., -1]
    
    
    # result = np.sqrt(w[...,-1]).T*v[...,-1].T
    result = np.sqrt(w).T*v.T
    # First axis is now antenna.. refer all phases to reference ant
    phi = np.angle(result[ref])
    amp = np.abs(result[ref])
    fac = (np.cos(phi) - 1.0j*np.sin(phi)) * (amp>0.0)
    result = (result*fac).T 
    # TODO try to reduce number of transposes
    outdims = list(range(axis)) + [-1, ] + list(range(axis, ndim-1))
    return result.transpose(outdims)

def gaincal_gpu(data, nant, ant_indices, axis=0, ref=0, avg=[], nit=3):
    """Derives amplitude/phase calibration factors from the data array
    for the given baseline axis.  In the returned array, the baseline
    dimension is converted to antenna.  No other axes are modified.
    Note this internally makes a transposed copy of the data so be
    careful with memory usage in the case of large data sets.  A list
    of axes to average over before solving can be given in the avg
    argument (length-1 dimensions are kept so that the solution can be
    applied to the original data)."""

    nbl = data.shape[axis]
    ndim = len(data.shape)
    #(check, nant) = bl2ant(nbl)
    #print(nant)
    #if check != 0:
    #    raise RuntimeError("Specified axis dimension (%d) is not a valid number of baselines" % nbl)
    if avg != []:
        # Average, ignoring zeros
        #norm = np.count_nonzero(data,axis=avg,keepdims=True) # requires numpy 1.19 for keepdims
        norm = np.count_nonzero(data,axis=tuple(avg))
        norm[np.where(norm==0)] = 1
        data = data.sum(axis=tuple(avg), keepdims=True)
        norm = norm.reshape(data.shape) # workaround lack of keepdims
        data = data / norm
    
    
    print("Making a reordered visiility set")
    tdata = np.zeros(data.shape[:axis]+data.shape[axis+1:]+(nant, nant),
                     dtype=data.dtype)
    for i in range(nbl):
        #(a0, a1) = bl2ant(i)
        #print(a0, a1)
        [a0, a1] = ant_indices[i]
        if a0 != a1:
            tdata[..., a0, a1] = data.take(i, axis=axis)
            tdata[..., a1, a0] = np.conj(data.take(i, axis=axis))
    
    #Transferring the array to GPU
    print("Transferring data to GPU")
    with cp.cuda.Device(0):
        tdata_gpu = cp.asarray(tdata)
    
    print("Eigen value decomposition")

    for it in range(nit):
        (wtmp, vtmp) = linalg_gpu.eigh(tdata_gpu)
        v = vtmp[..., -1].copy()
        w = wtmp[..., -1]
        for i in range(nant):
            tdata_gpu[..., i, i] = w*(v.real[..., i]**2 + v.imag[..., i]**2)
           


    #(wtmp, vtmp) = linalg.eigh(tdata)
    #v = vtmp[..., -1].copy()
    #w = wtmp[..., -1]
    del tdata_gpu
    
    print("Calculating the gain")
    # result = np.sqrt(w[...,-1]).T*v[...,-1].T
    result = cp.sqrt(w).T*v.T
    # First axis is now antenna.. refer all phases to reference ant
    phi = cp.angle(result[ref])
    amp = cp.abs(result[ref])
    fac = (cp.cos(phi) - 1.0j*cp.sin(phi)) * (amp>0.0)
    result = (result*fac).T 

    result_cpu = result.get()

    # TODO try to reduce number of transposes
    outdims = list(range(axis)) + [-1, ] + list(range(axis, ndim-1))

    return result_cpu.transpose(outdims)





def applycal(data, caldata, nant_check, ant_indices, axis=0, phaseonly=False):
    """
    Apply the complex gain calibration given in the caldata array
    to the data array.  The baseline/antenna axis must be specified in
    the axis argument.  Dimensions of all other axes must match up
    (in the numpy broadcast sense) between the two arrays.

    Actually what matters is the correct list of antennas and not the number of antennas
    Solutions derived from the same set of antennas must be applied to the 
    another dataset which has same set of antennas. 
    """

    ndim = len(data.shape)
    nbl = data.shape[axis]
    nant = caldata.shape[axis]
    #(check, nant_check) = bl2ant(nbl)
    #if check != 0:
    #    raise RuntimeError("Specified axis dimension (%d) is not a valid number of baselines" % nbl)
    if nant != nant_check:
        raise RuntimeError("Number of antennas does not match (data=%d, caldata=%d)" % (nant_check, nant))
    if phaseonly:
        icaldata = np.abs(caldata)/caldata
    else:
        icaldata = 1.0/caldata
    icaldata[np.where(np.isfinite(icaldata) == False)] = 0.0j
    # Modifies data in place.  Would it be better to return a calibrated
    # copy instead of touching the original?
    for ibl in range(nbl):
        # Must be some cleaner way to do this..?
        dslice = (slice(None),)*axis + (ibl,) + (slice(None),)*(ndim-axis-1)
        [a1, a2] = ant_indices[ibl]
        calfac =  icaldata.take(a1, axis=axis) * icaldata.take(a2, axis=axis).conj()
        data[dslice] *= calfac




