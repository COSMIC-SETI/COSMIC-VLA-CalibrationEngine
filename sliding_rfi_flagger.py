import numpy as np
from scipy.stats import median_abs_deviation as mad


def flag_rfi_real(data, winSize, clip=3):
          
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


   
def flag_rfi_complex_pol(data, winSize, clip):
    """
    Flagging data from a dataset of the form [freqs, pols] or 
    a time averaged spectra for eech polarization from a baseline
    
    Returns :
    RFI affected channels for each polarization as a dictionary
    Smooth bandpass model for each polarization
    
    """     

    smth = data*0.0

    # Compute the smoothed bandpass model
    for i in range(smth.shape[0]):
        mn = int(max([0, i-winSize/2]))
        mx = int(min([i+winSize/2+1, smth.shape[0]]))
        smth[i,:] = np.median(data[mn:mx,:], axis = 0)      
           
    #Subtracting the model from the data, calculate the median and mad of the difference
    diff = data-smth 
    med = np.median(diff, axis = 0)
    sig_md = mad(diff, axis = 0)

    #For each pol, find data points deviate more than a threshold
    bad_chans = {}
    for pol in range(data.shape[1]):
        bad = np.argwhere(np.abs(diff[:,pol]-med[pol]) > clip*np.abs(sig_md[pol]))[:,0]
        bad_chans[pol] = bad

    #returns the bad_chans per pols and smooth bandpass model
    return  bad_chans, smth
