"""
Written by Savin Shynu Varghese
Scripts to derive delay and phase calibrations from UVH5 files using Pyuvdata
Also, produces variety of diagnostic plots
"""
import sys
import os
import argparse
import time
import json
import numpy as np
import pandas as pd
import redis
from matplotlib import pyplot as plt
import pyuvdata.utils as uvutils
from pyuvdata import UVData
from calib_util import gaincal_cpu, gaincal_gpu, applycal
from sliding_rfi_flagger import flag_rfi



def flag_spectrum(spectrum, win, threshold = 3):

    """
    Function to flag bad RFI channel using a 
    sliding median window:
    Can be used if the delay values derived does not makes any 
    sense
    """

    #Getting bad channels
    bad_chan = flag_rfi(spectrum, win, threshold)
    
    ##Zeroing bad channels
    spectrum[bad_chan[:,0]] = 0
    return spectrum




class calibrate_uvh5:

    def __init__(self, datafile):

        #Initializing the pyuvdata object and reading the files
        self.datafile = datafile
        self.uvd = UVData()
        self.uvd.read(datafile, fix_old_proj=False)
        self.metadata = self.get_metadata()
        self.vis_data = self.get_vis_data_new()
        self.ant_indices = self.get_ant_array_indices()
        self.redis_obj = redis.Redis(host="redishost", decode_responses=True)

    def get_metadata(self):
        """
        Reading the metadata from the uvh5 object 
        and adding them to a dictionary
        """

        nant_data = self.uvd.Nants_data
        nant_array = self.uvd.Nants_telescope
        ant_names = self.uvd.antenna_names
        ant_curr = self.uvd.get_ants()
        nfreqs = self.uvd.Nfreqs
        ntimes = self.uvd.Ntimes
        npols = self.uvd.Npols
        bls = self.uvd.Nbls
        nspws = self.uvd.Nspws
        chan_width = self.uvd.channel_width
        #make some changes here, each visibiliy could have a different integration time
        intg_time = self.uvd.integration_time[0]
        object_name = self.uvd.object_name.split('.')
        source = object_name[0]
        tuning = object_name[-1]
        telescope = self.uvd.telescope_name
        pol_array = uvutils.polnum2str(self.uvd.polarization_array)
        freq_array = self.uvd.freq_array[0,:]
        lobs = ntimes*intg_time
        
        time_array = np.arange(ntimes)*intg_time
        metadata  = {'nant_data' : nant_data,
        'nant_array' : nant_array,
        'ant_names' : ant_names, 
        'ant_curr' : ant_curr,
        'nfreqs' : nfreqs,
        'ntimes' : ntimes,
        'npols': npols,
        'nbls' : bls,
        'nspws' : nspws,
        'chan_width': chan_width,
        'intg_time' : intg_time,
        'lobs' : lobs,
        'source': source,
        'telescope' : telescope,
        'pol_array' : pol_array,
        'freq_array' : freq_array,
        'time_array' : time_array,
        'tuning' : tuning}
        return metadata

    def print_metadata(self):
        #Print out the observation details
        
        meta = self.metadata
        print(f" Observations from {meta['telescope']}: \n\
                Source observed: {meta['source']} \n\
                No. of time integrations: {meta['ntimes']} \n\
                Length of time integration: {meta['intg_time']} s \n\
                Length of observations: {meta['lobs']} s \n\
                No. of frequency channels: {meta['nfreqs']} \n\
                Width of frequency channel: {meta['chan_width']/1e+3} kHz\n\
                Start freq: {meta['freq_array'][0]/1e+6} MHz, Stop freq: {meta['freq_array'][-1]/1e+6} MHz \n\
                Observation bandwidth: {(meta['freq_array'][-1] - meta['freq_array'][0])/1e+6} MHz \n\
                No. of spectral windows: {meta['nspws']}  \n\
                Polarization array: {meta['pol_array']} \n\
                No. of polarizations: {meta['npols']}   \n\
                Data array shape: {self.vis_data.shape} \n\
                No. of baselines: {meta['nbls']}  \n\
                No. of antennas present in data: {meta['nant_data']} \n\
                Current antenna list in the data: {meta['ant_curr']} \n\
                No. of antennas in the array: {meta['nant_array']} \n\
                Antenna name: {meta['ant_names']} \n\
                Tuning: {meta['tuning']}")


    def get_uvw_data(self):
        #Get the UVW info
        uvw_array = self.uvd.uvw_array
        uvw_array = uvw_array.reshape(self.metadata['ntimes'], self.metadata['bls'], 3)
        return uvw_array

    """
    def get_vis_data(self):
        
        # This seems to be a straight forward method
        # But the values out of uvd_data_array does not make any sense


        vis = np.zeros((self.metadata['nbls'], self.metadata['ntimes'], self.metadata['nfreqs'], self.metadata['npols']), dtype = 'complex128')
        print(self.uvd.data_array.dtype)
        vis = self.uvd.data_array.copy()
        new_shape = (self.metadata['nbls'], self.metadata['ntimes'])+vis.shape[1:]
        return np.squeeze(vis.reshape(new_shape))
    """

    def get_vis_data_new(self):
        
        """
        Iterate baseline by baseline and collect the
        visibility data
        """
        ant1, ant2 = self.uvd.baseline_to_antnums(self.uvd.baseline_array[:self.metadata['nbls']])
        vis_new = np.zeros((self.metadata['nbls'], self.metadata['ntimes'], self.metadata['nfreqs'], self.metadata['npols']), dtype = 'complex128')
        for i in range(self.metadata['nbls']):
            vis_new[i,...] = self.uvd.get_data(ant1[i], ant2[i])

        return np.squeeze(vis_new)
       
    def write_ms(self, outdir):
        """
        Write the uvh5 data file into measurement set for CASA
        """
        outfile = os.path.join(outdir, os.path.basename(self.datafile).split('.')[0] +'.ms')
        return self.uvd.write_ms(outfile)


    def derive_phase(self, outdir):
        """
        Derive gains per antenna/channel/polarizations
        using some of the sdmpy calibration codes
        """

        print("Deriving Calibrations now")
        t1 = time.time()
        #Check the ref antenna here, make sure if it is antenna 10.
        gainsol_dict = gaincal_cpu(self.vis_data, self.metadata['ant_curr'], self.ant_indices,  axis = 0, avg = [1], ref_ant = 10)
        gain = np.squeeze(gainsol_dict['gain_val'])
        gain_ant = gainsol_dict['antennas']
        
        #for i in range(1,29):
        #    ant = "ea"+str(i).zfill(2)
        #    json_gain_dict['ant_gains'][ant] = {}

        json_gain_dict = {'gains':{}, 'freqs_hz': self.metadata['freq_array'].tolist()}
               
        
        
        #Let's go through each antenna in the gain antenna list and update the gain values
        for i, ant in enumerate(gain_ant):
            ant_str = "ea"+str(ant).zfill(2)
            json_gain_dict['gains'][ant_str] = {}
            json_gain_dict['gains'][ant_str]['gain_pol0_real'] = gain[i, :, 0].real.tolist()
            json_gain_dict['gains'][ant_str]['gain_pol0_imag'] = gain[i, :, 0].imag.tolist()
            json_gain_dict['gains'][ant_str]['gain_pol1_real'] = gain[i, :, 3].real.tolist()
            json_gain_dict['gains'][ant_str]['gain_pol1_imag'] = gain[i, :, 3].imag.tolist()
        
        

        #Writting the dictionary as a json file
        outfile_json = os.path.join(outdir, os.path.basename(self.datafile).split('.')[0]+ f"_gain_dict.json")

        print("Writing our the gains per antenna/freq/pols")
        with open(outfile_json, "w") as jh:
            json.dump(json_gain_dict, jh)

        t2 = time.time()
        print(f"Took {t2-t1}s for getting solution from {self.metadata['lobs']}s of data")

        print(f"Solution shape: {gainsol_dict['gain_val'].shape}")
        
        return json_gain_dict

    def apply_phase(self, gainsol):
        """
        Apply the derived gains to a UVH5 dataset. 
        Also the antenna list in gain has to match with the new dataset, otherwise applying the gains to the
        correlated matrix would be difficult
        """
        data_cp = self.vis_data.copy()
        applycal(data_cp, gain_dict, self.metadata['ant_curr'], self.ant_indices, axis=0, phaseonly=False)
        print(self.vis_data.dtype)
        return data_cp
    

    def get_ant_array_indices(self):

        """
        Getting baseline indices the way data is arranged in the 
        pyuvdata object
        """
        auto = []
        cross = []
        nant = self.metadata['nant_data']
        for i in range(nant):
            for j in range(nant):
                if i == j:
                    auto.append([i,j])

        for i in range(nant):
            for j in range(nant):
                if i < j:
                    cross.append([i,j])

        ant_indices = auto+cross        
        
        return ant_indices


    def plot_gain_phases_amp(self, gain_dict, outdir, plot_amp = False):

        """
        Plots the amplitude and phase (averaged over time) across frequency for a gain solutions (antenna, times, frequency, pols)
        """
        
        print("plotting gain phase & amp vs freq ")
        
        gain_ant= gain_dict['antennas']
        gain = gain_dict['gain_val']

        gain_avg = np.squeeze(np.mean(gain, axis=1))
        nant = len(gain_ant)

        grid_x = 6
        grid_y = 5
        grid_val = grid_x*grid_y
        nplts = int(np.ceil(nant/(grid_val)))
        
        
        for n in range(nplts):
            
            outfile = os.path.join(outdir, os.path.basename(self.datafile).split('.')[0]+ f"gain_phase_freq_{n}.png")
            fig, axs = plt.subplots(grid_x, grid_y, sharex  = True, sharey = True, constrained_layout=True, figsize = (12,12))
            for i in range(grid_x):
                for j in range(grid_y):
                    rbl = (i*grid_y)+j
                    bl = grid_val*n + rbl
                    #print(bl)
                    if bl < nant:

                        #Picking the antenna
                        gain_rr = gain_avg[bl,:,0]
                        gain_ll = gain_avg[bl,:,3]

                        axs[i,j].plot(self.metadata['freq_array']/1e+9, np.angle(gain_rr, deg = True), '.',  label = "RR")
                        axs[i,j].plot(self.metadata['freq_array']/1e+9, np.angle(gain_ll, deg = True), '.',  label = "LL")

                        axs[i,j].set_title(f"ea{gain_ant[bl]}")                    
                        axs[i,j].legend(loc = 'upper right')
            
            fig.suptitle("Gain: Phase vs Freq (averaged in time), RR, LL")
            fig.supylabel("Phase (degrees)")
            fig.supxlabel("Frequency (GHz)")
            plt.savefig(outfile, dpi = 150)
            plt.close()
        
        
        if plot_amp:

            #plotting the amplitude 
            print("Plotting Gain amplitude vs freq over time")
            for n in range(nplts):
                 
                outfile = os.path.join(outdir, os.path.basename(self.datafile).split('.')[0]+ f"gain_amp_vs_freq_{n}.png")
                fig, axs = plt.subplots(grid_x, grid_y, sharex  = True, sharey = True, constrained_layout=True, figsize = (12,12))
                for i in range(grid_x):
                    for j in range(grid_y):
                        rbl = (i*grid_y)+j
                        bl = grid_val*n + rbl
                        if bl < nant:

                            #Picking the antenna
                            gain_rr = gain_avg[bl,:,0]
                            gain_ll = gain_avg[bl,:,3]

                            axs[i,j].plot(self.metadata['freq_array']/1e+9, np.abs(gain_rr), '.',  label = "RR")
                            axs[i,j].plot(self.metadata['freq_array']/1e+9, np.abs(gain_ll), '.',  label = "LL")

                            axs[i,j].set_title(f"ea{gain_ant[bl]}")
                            axs[i,j].legend(loc = 'upper right')
           
                
                fig.suptitle("Gain: Amplitude vs Freq (averaged in time), RR, LL")
                fig.supylabel("Amplitude (a.u.)")
                fig.supxlabel("Frequency (GHz)")
                plt.savefig(outfile, dpi = 150)
                plt.close()

    def plot_phases_vs_freq(self, data, outdir, plot_amp = False, corrected = False):
        
        """
        Plotting the phase and amplitude across frequency for a visibility dataset
        Use corrected = True to adjust the title after the gain corrections
        """
        print("plotting phase vs freq on all baselines")

        data_avg = np.mean(data, axis=1)
        ant1, ant2 = self.uvd.baseline_to_antnums(self.uvd.baseline_array[:self.metadata['nbls']])
        
        nbls = self.metadata['nbls']
        
        grid_x = 6
        grid_y = 6
        grid_val = grid_x*grid_y
        nplts = int(np.ceil(nbls/(grid_val)))
        
        
        for n in range(nplts):
            if not corrected:
                outfile = os.path.join(outdir, os.path.basename(self.datafile).split('.')[0]+ f"_uncor_phase_freq_{n}.png")
            else:
                outfile = os.path.join(outdir, os.path.basename(self.datafile).split('.')[0]+ f"_cor_phase_freq_{n}.png")
            fig, axs = plt.subplots(grid_x, grid_y, sharex  = True, sharey = True, constrained_layout=True, figsize = (12,12))
            for i in range(grid_x):
                for j in range(grid_y):
                    rbl = (i*grid_y)+j
                    bl = grid_val*n + rbl
                    if bl < nbls:

                        #Picking the baseline
                        data_bls_rr = data_avg[bl,:,0]
                        data_bls_ll = data_avg[bl,:,3]

                        axs[i,j].plot(self.metadata['freq_array']/1e+9, np.angle(data_bls_rr, deg = True), '.',  label = "RR")
                        axs[i,j].plot(self.metadata['freq_array']/1e+9, np.angle(data_bls_ll, deg = True), '.',  label = "LL")

                        axs[i,j].set_title(f"ea{ant1[bl]} - ea{ant2[bl]}")
                        axs[i,j].legend(loc = 'upper right')
            if not corrected:
                fig.suptitle("Uncorrected: Phase vs Freq (averaged in time), RR, LL")
            else:
                fig.suptitle("Corrected: Phase vs Freq (averaged in time), RR, LL")
            fig.supylabel("Phase (degrees)")
            fig.supxlabel("Frequency (GHz)")
            plt.savefig(outfile, dpi = 150)
            plt.close()
        
        
        if plot_amp:
            #plotting the amplitude 
            print("Plotting amplitude vs freq over time")
            for n in range(nplts):
                if not corrected:
                    outfile = os.path.join(outdir, os.path.basename(self.datafile).split('.')[0]+ f"_uncor_amp_vs_freq_{n}.png")
                else:   
                    outfile = os.path.join(outdir, os.path.basename(self.datafile).split('.')[0]+ f"_cor_amp_vs_freq_{n}.png")
                fig, axs = plt.subplots(grid_x, grid_y, sharex  = True, sharey = True, constrained_layout=True, figsize = (12,12))
                for i in range(grid_x):
                    for j in range(grid_y):
                        rbl = (i*grid_y)+j
                        bl = grid_val*n + rbl
                        if bl < nbls:

                            #Picking the baseline
                            data_bls_rr = data_avg[bl,:,0]
                            data_bls_ll = data_avg[bl,:,3]

                            axs[i,j].plot(self.metadata['freq_array']/1e+9, np.abs(data_bls_rr), '.',  label = "RR")
                            axs[i,j].plot(self.metadata['freq_array']/1e+9, np.abs(data_bls_ll), '.',  label = "LL")

                            axs[i,j].set_title(f"ea{ant1[bl]} - ea{ant2[bl]}")
                            axs[i,j].legend(loc = 'upper right')
           
                if not corrected:
                    fig.suptitle("Uncorrected: Amplitude vs Freq (averaged in time), RR, LL")
                else:
                    fig.suptitle("Corrected: Amplitude vs Freq (averaged in time), RR, LL")
                fig.supylabel("Amplitude (a.u.)")
                fig.supxlabel("Frequency (GHz)")
                plt.savefig(outfile, dpi = 150)
                plt.close()
            
    def plot_phases_waterflall(self, data, outdir, track_phase = False):

        """
        Make waterfall plots of phases from the visibility dataset
        """
       
        data = np.squeeze(data)
        ant1, ant2 = self.uvd.baseline_to_antnums(self.uvd.baseline_array[:self.metadata['nbls']])
        
        nbls = self.metadata['nbls']
        times_ar = self.metadata['time_array']
        #grid = int(np.ceil(np.sqrt(nbls)))
        grid_x = 6
        grid_y = 6
        grid_val = grid_x*grid_y
        nplts = int(np.ceil(nbls/(grid_val)))

        dely = self.metadata['time_array'][1] - self.metadata['time_array'][0]
        yr = np.linspace(self.metadata['time_array'].min()-dely/2.0, self.metadata['time_array'].max()+dely/2.0, len(self.metadata['time_array'])+1)
        delx = self.metadata['freq_array'][1] - self.metadata['freq_array'][0]
        xr = np.linspace((self.metadata['freq_array'].min()-(delx/2.0))/1e+9, (self.metadata['freq_array'].max() + (delx/2.0))/1e+9, len(self.metadata['freq_array'])+1)

        #Plotting the RR
        print("Plotting phase vs freq over time for RR")
        for n in range(nplts):
            outfile_rr = os.path.join(outdir, os.path.basename(self.datafile).split('.')[0]+ f"_phase_waterfall_rr_{n}.png")
            fig_ph1, axs_ph1 = plt.subplots(grid_x, grid_y, sharex  = True, sharey = True, constrained_layout=True, figsize = (12,12))
            for i in range(grid_x):
                for j in range(grid_y):
                    rbl = (i*grid_y)+j
                    bl = grid_val*n + rbl
                    if bl < nbls:

                        #Picking the baseline
                        data_bls_rr = data[bl,:,:,0]

                        axs_ph1[i,j].pcolormesh(xr, yr, np.angle(data_bls_rr, deg = True))
                        axs_ph1[i,j].set_title(f"ea{ant1[bl]} - ea{ant2[bl]}")
           
            fig_ph1.suptitle("Phase vs Freq over time, RR")
            fig_ph1.supylabel("Time (s)")
            fig_ph1.supxlabel("Frequency (GHz)")
            plt.savefig(outfile_rr, dpi = 150)
            plt.close()
        
    
        #plotting the LL 
        print("Plotting phase vs freq over time for LL")
        for n in range(nplts):
            outfile_ll = os.path.join(outdir, os.path.basename(self.datafile).split('.')[0]+ f"_phase_waterfall_ll_{n}.png")
            fig_ph2, axs_ph2 = plt.subplots(grid_x, grid_y, sharex  = True, sharey = True, constrained_layout=True, figsize = (12,12))
            for i in range(grid_x):
                for j in range(grid_y):
                    rbl = (i*grid_y)+j
                    bl = grid_val*n + rbl
                    if bl < nbls:

                        #Picking the baseline
                        data_bls_ll = data[bl,:,:,3]

                        axs_ph2[i,j].pcolormesh(xr, yr, np.angle(data_bls_ll, deg = True))
                        axs_ph2[i,j].set_title(f"ea{ant1[bl]} - ea{ant2[bl]}")
           
            fig_ph2.suptitle("Phase vs Freq over time, LL")
            fig_ph2.supylabel("Time (s)")
            fig_ph2.supxlabel("Frequency (GHz)")
            plt.savefig(outfile_ll, dpi = 150)
            plt.close()

        if track_phase:
            print("Plotting averaged phase over frequency vs  time")
            for n in range(nplts):
                outfile_ph_track = os.path.join(outdir, os.path.basename(self.datafile).split('.')[0]+ f"_phase_tracked_rr_ll_{n}.png")
                fig_ph3, axs_ph3 = plt.subplots(grid_x, grid_y, sharex  = True, sharey = True, constrained_layout=True, figsize = (12,12))
                for i in range(grid_x):
                    for j in range(grid_y):
                        rbl = (i*grid_y)+j
                        bl = grid_val*n + rbl
                        if bl < nbls:

                            #Picking the baseline
                            data_bls_rr = data[bl,:,:,0]
                            data_bls_ll = data[bl,:,:,3]
                            
                            ph_rr = np.angle(np.mean(data_bls_rr, axis = 1), deg = True)
                            ph_ll = np.angle(np.mean(data_bls_ll, axis = 1), deg = True)

                            
                            axs_ph3[i,j].plot(times_ar, ph_rr , '.', label = 'RR')
                            axs_ph3[i,j].plot(times_ar, ph_ll , '.', label = 'LL')    
                            #axs_d4[i,j].set_ylim(-50,50)
                            axs_ph3[i,j].set_title(f"ea{ant1[bl]} - ea{ant2[bl]}")
                            axs_ph3[i,j].legend(loc = 'upper right')

           
                fig_ph3.suptitle(f"Phase averaged (over frequency) vs time")
                fig_ph3.supxlabel("Time (s)")
                fig_ph3.supylabel("Phase averaged over frequency (degrees) ")
                plt.savefig(outfile_ph_track, dpi = 150)
                plt.close()   


    def get_res_delays(self, data, outdir, ref_ant = 'ea10'):

        data = np.squeeze(data) # removing redundant axis
        fin_nchan =  1024 # Getting frequency shape

        #Defining  total frequency channels and fine channel bandwidths in Hz to get the time lags
        tlags = np.fft.fftfreq(fin_nchan, self.metadata['chan_width'])
        tlags = np.fft.fftshift(tlags)*1e+9 #Converting the time lag into ns

        #Antenna corresponding to each baselines
        ant1, ant2 = self.uvd.baseline_to_antnums(self.uvd.baseline_array[:self.metadata['nbls']])
        
        #Total number of baselines
        nbls = self.metadata['nbls']

        #Time array
        times_ar = self.metadata['time_array']

        #Array to store the delay values across time for RR and LL
        delay_vals = np.zeros((nbls, len(times_ar),2),  dtype = 'float32')

        # Writing the delay values to a csv file
        #Opening a file to save the delays for each baselines

        # try:
        #     tun_mnt = self.datafile.split('/')[2]
        #     if tun_mnt == 'buf0':
        #         tun = 'AC'
        #     else:
        #         tun = 'BD'
        # except:
        #     tun = 'Unknown'
        tun = self.metadata['tuning']
        outfile_res = os.path.join(outdir, os.path.basename(self.datafile).split('.')[0]+ f"_res_delay_{tun}.csv")
        dh = open(outfile_res, "w")

        dh.write(",".join(
                [
                "Baseline",
                "res_pol0",
                "res_pol1"
                ]
                )+"\n")
        
        for bl in range(nbls):
            #Picking the RR data
            data_bls_rr = data[bl,:,:,0]

            #Conduct an ifft along the frequency axis
            data_bls_rr_ifft = np.fft.ifft(data_bls_rr, n = fin_nchan, axis = 1)
            data_bls_rr_ifft = np.fft.fftshift(data_bls_rr_ifft, axes = 1)

            spec_rr = np.abs(data_bls_rr_ifft)
            peak_inds = np.argmax(spec_rr, axis = 1)
            delay_vals[bl,:,0] = tlags[peak_inds]  

            #Picking the LL data
            data_bls_ll = data[bl,:,:,3]
                        
            #Conduct an ifft along the frequency axis
            data_bls_ll_ifft = np.fft.ifft(data_bls_ll, n = fin_nchan, axis = 1)
            data_bls_ll_ifft = np.fft.fftshift(data_bls_ll_ifft, axes = 1)
                        
            spec_ll = np.abs(data_bls_ll_ifft)
            peak_inds = np.argmax(spec_ll, axis = 1)
            delay_vals[bl,:,1] = tlags[peak_inds]


            #writing the delay value part, taking a mean across time assuming delay are constant
            ant1_str = 'ea' + str(ant1[bl]).zfill(2)
            ant2_str = 'ea' + str(ant2[bl]).zfill(2)
            #bls_str = 'ea'+ ant1_str +'-ea'+ ant2_str
            ant_base = [ant1_str, ant2_str]
            #print(ant_base)
            if ref_ant in ant_base:
                ant_base.remove(ref_ant)
                ant_new = ant_base[0]
                #print(ant_new)
                if ant1_str == ref_ant and ant2_str == ref_ant:
                   #if both antennas are the ref antenna case
                    dh.write(f"{ant_new}, {np.mean(delay_vals[bl,:,0])}, {np.mean(delay_vals[bl,:,1])} \n")
                    continue
                if ant1_str == ref_ant:
                    #Negating the delay values so that all the delay values are with reference to ref ant, ea_x - ea_ref, if not make the sign negative
                    dh.write(f"{ant_new}, {-np.mean(delay_vals[bl,:,0])}, {-np.mean(delay_vals[bl,:,1])} \n")
                if ant2_str == ref_ant:
                    dh.write(f"{ant_new}, {np.mean(delay_vals[bl,:,0])}, {np.mean(delay_vals[bl,:,1])} \n")
            

        dh.close()
        return outfile_res

    def get_phases(self, ref_ant = 'ea10'):

        nbl, ntime, nchan, npol = self.vis_data.shape
        nant = self.metadata['nant_data']

        #Antenna corresponding to each baselines
        ant1, ant2 = self.uvd.baseline_to_antnums(self.uvd.baseline_array[:])
        
        ##Array to store the accumulated spectra for each antenna vs reference
        #spectra = {}
        #for ant in self.metadata['ant_names']:
        #    spectra[ant] = np.zeros([nchan, npol], dtype=complex)
        phase_vals = np.zeros([nant, 2, nchan], dtype=float)
        ant_names = ['' for _ in range(nant)]
        n = 0

        for bl in range(nbl):
            # only use baselines with a refant
            antname1 = 'ea%.2d' % ant1[bl]
            antname2 = 'ea%.2d' % ant2[bl]
            if (antname1 == ref_ant):
                calant = antname2
                flip = 1
            elif (antname2 == ref_ant):
                calant = antname1
                flip = -1
            else:
                # Skip baselines which don't include ref
                continue

            # average over time (assume tracking is working)
            data = self.vis_data[bl].mean(axis=0)

            phase_vals[n,0] = flip*np.angle(data[:,0])
            phase_vals[n,1] = flip*np.angle(data[:,3])
            ant_names[n] = calant
            n += 1
        return ant_names, phase_vals


    def plot_delays_waterflall(self, data, outdir, track_delay = True):
        
        data = np.squeeze(data) # removing redundant axis
        
        fin_nchan = 1024 # Getting frequency shape

        #Defining  total frequency channels and fine channel bandwidths in Hz to get the time lags
        tlags = np.fft.fftfreq(fin_nchan, self.metadata['chan_width'])
        tlags = np.fft.fftshift(tlags)*1e+9 #Converting the time lag into ns

        #Antenna corresponding to each baselines
        ant1, ant2 = self.uvd.baseline_to_antnums(self.uvd.baseline_array[:self.metadata['nbls']])
        
        #Total number of baselines
        nbls = self.metadata['nbls']
        times_ar = self.metadata['time_array']
        
        #Storing delay values to track across time
        delay_vals = np.zeros((nbls, len(times_ar),2),  dtype = 'float32')

        grid_x = 6
        grid_y = 6
        grid_val = grid_x*grid_y
        nplts = int(np.ceil(nbls/(grid_val)))
        
        dely = self.metadata['time_array'][1] - self.metadata['time_array'][0]
        yr = np.linspace(self.metadata['time_array'].min()-dely/2.0, self.metadata['time_array'].max()+dely/2.0, len(self.metadata['time_array'])+1)
       
        delx = tlags[1] - tlags[0]
        xr = np.linspace(tlags.min()-(delx/2.0), tlags.max() + (delx/2.0), len(tlags)+1)

        #Plotting the RR delay waterfall
        print("Plotting delay vs time-lags over time for RR")
        for n in range(nplts):
            outfile_rr = os.path.join(outdir, os.path.basename(self.datafile).split('.')[0]+ f"_delay_waterfall_rr_{n}.png")
            fig_d1, axs_d1 = plt.subplots(grid_x, grid_y, sharex  = True, sharey = True, constrained_layout=True, figsize = (12,12))
            for i in range(grid_x):
                for j in range(grid_y):
                    rbl = (i*grid_y)+j
                    bl = grid_val*n + rbl
                    if bl < nbls:

                        #Picking the baseline
                        data_bls_rr = data[bl,:,:,0]

                        #Conduct an ifft along the frequency axis
                        data_bls_rr_ifft = np.fft.ifft(data_bls_rr, n = fin_nchan, axis = 1)
                        data_bls_rr_ifft = np.fft.fftshift(data_bls_rr_ifft, axes = 1)

                        spec = np.abs(data_bls_rr_ifft)
                        
                        peak_inds = np.argmax(spec, axis = 1)
                        delay_vals[bl,:,0] = tlags[peak_inds]
                        
                        
                        axs_d1[i,j].pcolormesh(xr, yr, spec)
                        
                        axs_d1[i,j].set_title(f"ea{ant1[bl]} - ea{ant2[bl]}")
           
            fig_d1.suptitle("Delay vs time-lags over time, RR")
            fig_d1.supylabel("Time (s)")
            fig_d1.supxlabel("Time-lags (ns)")
            plt.savefig(outfile_rr, dpi = 150)
            plt.close()
        

        #plotting the delay waterfall over time for each baseline
        print("Plotting delays vs time-lags over time for LL")
        for n in range(nplts):
            outfile_ll = os.path.join(outdir, os.path.basename(self.datafile).split('.')[0]+ f"_delay_waterfall_ll_{n}.png")
            fig_d2, axs_d2 = plt.subplots(grid_x, grid_y, sharex  = True, sharey = True, constrained_layout=True, figsize = (12,12))
            for i in range(grid_x):
                for j in range(grid_y):
                    rbl = (i*grid_y)+j
                    bl = grid_val*n + rbl
                    if bl < nbls:

                        #Picking the baseline
                        data_bls_ll = data[bl,:,:,3]
                        
                        #Conduct an ifft along the frequency axis
                        data_bls_ll_ifft = np.fft.ifft(data_bls_ll, n = fin_nchan, axis = 1)
                        data_bls_ll_ifft = np.fft.fftshift(data_bls_ll_ifft, axes = 1)
                        
                        spec = np.abs(data_bls_ll_ifft)
                        
                        peak_inds = np.argmax(spec, axis = 1)
                        delay_vals[bl,:,1] = tlags[peak_inds]
                       
                        axs_d2[i,j].pcolormesh(xr, yr, spec)
                        
                        axs_d2[i,j].set_title(f"ea{ant1[bl]} - ea{ant2[bl]}")
           
            fig_d2.suptitle("Delay vs time-lags over time, LL")
            fig_d2.supylabel("Time (s)")
            fig_d2.supxlabel("Time-lags (ns)")
            plt.savefig(outfile_ll, dpi = 150)
            plt.close()

        #plotting the delay values
        if track_delay:
            
            print("Plotting delay peaks vs time-lags over time")
            for n in range(nplts):
                outfile_peak = os.path.join(outdir, os.path.basename(self.datafile).split('.')[0]+ f"_delay_tracked_rr_ll_{n}.png")
                fig_d3, axs_d3 = plt.subplots(grid_x, grid_y, sharex  = True, sharey = True, constrained_layout=True, figsize = (12,12))
                for i in range(grid_x):
                    for j in range(grid_y):
                        rbl = (i*grid_y)+j
                        bl = grid_val*n + rbl
                        if bl < nbls:
                            
                            axs_d3[i,j].plot(times_ar, delay_vals[bl,:,0], '.', label = 'RR')
                            axs_d3[i,j].plot(times_ar, delay_vals[bl,:,1], '.', label = 'LL')    
                            #axs_d3[i,j].set_ylim(-50,50)
                            axs_d3[i,j].set_title(f"ea{ant1[bl]} - ea{ant2[bl]}")
                            axs_d3[i,j].legend(loc = 'upper right')

           
                fig_d3.suptitle(f"Delay peaks vs time, delay resolution: {round(tlags[1] - tlags[0], 3)} ns")
                fig_d3.supxlabel("Time (s)")
                fig_d3.supylabel("Delay peak (ns)")
                plt.savefig(outfile_peak, dpi = 150)
                plt.close()    
            
    
    def pub_to_redis(self, phase_outfile = None, delays_outfile = None):
        #create channel pubsub object for broadcasting changes to phases/residual-delays
        pubsub = self.redis_obj.pubsub(ignore_subscribe_messages=True)
        if phase_outfile is not None:
            try:
                pubsub.subscribe("gpu_calibrationphases")
            except redis.RedisError:
                raise redis.RedisError("""Unable to subscribe to gpu_calibrationphases channel to notify of 
                changes to GPU_calibrationPhases changes.""")
            with open(phase_outfile) as f:
                phase_dict = json.load(f)
            ant_names = phase_dict['ant_names']
            phase_vals_0 = phase_dict['phases_pol0']
            phase_vals_1 = phase_dict['phases_pol1']

            dict_to_pub = {}
            for i, ant in enumerate(ant_names):
                dict_to_pub[ant] = {
                    'freq_array' : phase_dict['freqs_hz'],
                    'pol0_phases' : phase_vals_0[i],
                    'pol1_phases' : phase_vals_1[i]
                }
            self.redis_obj.hset("GPU_calibrationPhases", str(self.metadata['freq_array'][0]/1e+6), json.dumps(dict_to_pub))
            self.redis_obj.publish("gpu_calibrationphases", json.dumps(True))

        if delays_outfile is not None:
            try:
                pubsub.subscribe("gpu_calibrationdelays")
            except redis.RedisError:
                raise redis.RedisError("""Unable to subscribe to gpu_calibrationdelays channel to notify of 
                changes to GPU_calibrationDelays changes.""")

            residual_delays = pd.read_csv(delays_outfile).to_dict('records')
            dict_to_pub = {}
            for i in range(len(residual_delays)):
                dict_to_pub[residual_delays[i]['Baseline']] = {
                    'freq_array' : self.metadata['freq_array'].tolist(),
                    'pol0_residual' : residual_delays[i]['res_pol0'],
                    'pol1_residual' : residual_delays[i]['res_pol1']
                }
            self.redis_obj.hset("GPU_calibrationDelays", str(self.metadata['freq_array'][0]/1e+6), json.dumps(dict_to_pub))
            self.redis_obj.publish("gpu_calibrationdelays", json.dumps(True))
    

def main(args):
    
    outfile_phase, outfile_delays = (None, None)

    # Creating an object with the input data file from solutions needed to be derived
    cal_ob = calibrate_uvh5(args.dat_file)
    
    #Print the metdata of the input file
    cal_ob.print_metadata()
    metadata = cal_ob.get_metadata()

    #Uncomment the following lines depending on the tasks to be completed

    #++++++++++++++++++++++++++++++++++++++++++++++++
    #Use if needed to convert file to a CASA MS format
    #cal_ob.write_ms(args.out_dir)

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Make a bunch of diagnostic plots before applying calibrations
    
    #plot the ampilitude and phase of visibility data
    #cal_ob.plot_phases_vs_freq(cal_ob.vis_data, args.out_dir, plot_amp = True)
    
    #plot the Phase waterfall plots of the visibility
    #cal_ob.plot_phases_waterflall(cal_ob.vis_data, args.out_dir, track_phase = True)

    #plot the Delay waterfall plots of the visibility
    #cal_ob.plot_delays_waterflall(cal_ob.vis_data, args.out_dir, track_delay = True)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    #Calculate the delays and spit out the delay values per baseline in the out_dir

    if args.gendelay:
        outfile_delays = cal_ob.get_res_delays(cal_ob.vis_data, args.out_dir)
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Derive the gain solutions from the visibility data
    
    #The gain dictinary obtained from sdmpy
    # Contains the list of antennas, ref antenna used to derive gain and the gain solutions in the form of (nant, ntimes, nfreqs, pols)
    if args.gengain:

        gdict = cal_ob.derive_phase(args.out_dir)
        #print(gdict)

    if args.genphase:
        #gain = cal_ob.derive_phase() # An antenna x time x channel x ?cross-pol?
        ## Assume we have legit phase tracking and little transient RFI
        #gain_av = gain.mean(axis=1) # average_over_time
        #out = {
        #    'ant_names': metadata['ant_names'],
        #    'freqs_hz': metadata['freq_array'].tolist(),
        #    'phases_pol0': np.angle(gain_av[:,:,0]).tolist(),
        #    'phases_pol1': np.angle(gain_av[:,:,3]).tolist(),
        #}
        antnames, phases = cal_ob.get_phases() # An antenna x time x channel x ?cross-pol?
        out = {
            'ant_names': antnames,
            'freqs_hz': metadata['freq_array'].tolist(),
            'phases_pol0': phases[:,0].tolist(),
            'phases_pol1': phases[:,1].tolist(),
        }
        outfile_phase = os.path.join(args.out_dir, os.path.basename(args.dat_file) + '_phasecal.json')
        with open(outfile_phase, 'w') as fh:
            json.dump(out, fh)
            
    if args.pub_to_redis:
        cal_ob.pub_to_redis(phase_outfile = outfile_phase, delays_outfile = outfile_delays)
    #Plotting amplitude and phase of the gain solutions
    #cal_ob.plot_gain_phases_amp(gain, args.out_dir, plot_amp = True)

    #Apply the solutions to the same dataset and plot the phases and amplitudes
    #cal_data = cal_ob.apply_phase(gain)
    #cal_ob.plot_phases_vs_freq(cal_data, args.out_dir, plot_amp = True, corrected = True)
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++==
    
    #Apply the solutions to a different dataset
    #In that case a create a different object of the same class withe the apply_dat_file
    # Creating an object with the datset to apply the solutions, apply the solutions and plot the phase and amp
    #cal_apply_ob = calibrate_uvh5(args.apply_dat_file)
    #cal_data_apply = cal_apply_ob.apply_phase(gain_dict) #Gain derived from a different file
    #cal_apply_ob.plot_phases_vs_freq(cal_data_apply, args.out_dir, plot_amp = True, corrected = True)
    
    
if __name__ == '__main__':
    
    # Argument parser taking various arguments
    parser = argparse.ArgumentParser(
        description='Reads UVH5 files, derives delay and gain calibrations, apply to the data, make a bunch of diagnostic plots',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--dat_file', type = str, required = True, help = 'UVH5 file to derive delay and phase calibrations')
    parser.add_argument('-ad','--apply_dat_file', type = str, required = False, help = 'UVH5 file to apply solutions derived from UVH5 file')
    parser.add_argument('-o','--out_dir', type = str, required = True, help = 'Output directory to save the plots')
    parser.add_argument('--gengain', action='store_true',
            help = 'If set, generate a json file of output gain per antenna/freq/pol')
    parser.add_argument('--genphase', action='store_true',
            help = 'If set, generate a file of output phases per antpol')
    parser.add_argument('--gendelay', action='store_true',
            help = 'If set, generate a file of output delays per antpol')
    parser.add_argument('--pub-to-redis', action="store_true", help ="Set up a redis object and publish the residual delays and calibration phases to it.")

    args = parser.parse_args()
    main(args)



