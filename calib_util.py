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




def gaincal_cpu(data, ant_curr, ant_indices, axis=1, ref_ant=10, avg=[], nit=3):
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
    nant = len(ant_curr)

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
    

    result = np.sqrt(w).T*v.T
    # refer all phases to reference ant, find it from the list
    if ref_ant in ant_curr:
        #Finding the index of ref antenna
        ref_ind = np.argwhere((ant_curr == ref_ant))[0]
        #print(ant_curr[ref_ind])
    else:
        print(f"The given reference antenna not in the current list of antennas, so choosing the first antenna (ea{ant_curr[0]})  in the list")
        ref_ind = 0
        ref_ant = ant_curr[ref_ind]
    
    phi = np.angle(result[ref_ind])
    amp = np.abs(result[ref_ind])
    fac = (np.cos(phi) - 1.0j*np.sin(phi)) * (amp>0.0)
    result = (result*fac).T 

    # TODO try to reduce number of transposes
    outdims = list(range(axis)) + [-1, ] + list(range(axis, ndim-1))
    gain = result.transpose(outdims)

    gain_dict = {'antennas': ant_curr, 'ref_antenna': ref_ant, 'gain_val': gain}
    return gain_dict

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





def applycal(data, gain_dict, ant_list, ant_indices, axis=0, phaseonly=False):
    """
    Apply the complex gain calibration given in the caldata array
    to the data array.  The baseline/antenna axis must be specified in
    the axis argument.  Dimensions of all other axes must match up
    (in the numpy broadcast sense) between the two arrays.

    Actually what matters is the correct list of antennas and not the number of antennas
    Solutions derived from the same set of antennas must be applied to the 
    another dataset which has same set of antennas. 
    """
    #Antenna list in the gain values
    ant_gain  = gain_dict['antennas']
    
    #Gain values
    caldata = gain_dict['gain_val']

    ndim = len(data.shape)
    nbl = data.shape[axis]
    nant = caldata.shape[axis]
    
    #Here you can apply the correction to the gain dataset to another visibility dataset
    #only if both of them has the same set of antennas. Let's do that check here

    default_val = True
    if len(ant_gain) != len(ant_list):
        raise RuntimeError(f"Number of antennas in gain ({len(ant_gain)}) does not match the dataset to be applied ({len(ant_list)})")
    else:
        for i, ant1 in enumerate(ant_gain):
            ant2 = ant_list[i]
            if ant1 !=  ant2:
                default_val = False
    
        if not default_val:
            raise RuntimeError("The antennas in gain solution and dataset are different")
    #print(default_val)
    
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
        #print(data[dslice].shape, calfac.shape)
        data[dslice] *= calfac




