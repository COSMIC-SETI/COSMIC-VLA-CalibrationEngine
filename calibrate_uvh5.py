"""
Written by Savin Shynu Varghese
Scripts to flag rfi, derive delay and phase calibrations from UVH5 files using Pyuvdata
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
import shutil
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues, redis_publish_dict_to_hash
from matplotlib import pyplot as plt
import pyuvdata.utils as uvutils
from pyuvdata import UVData
from calib_util import gaincal_cpu, applycal, flag_complex_vis_medf, calc_gain_grade
from sliding_rfi_flagger import flag_rfi_real

BAD_REFANT = []#["ea05","ea06"]

def flag_spectrum(spectrum, win, threshold = 3):

    """
    Function to flag bad RFI channel using a 
    sliding median window:
    Can be used if the delay values derived does not makes any 
    sense
    """

    #Getting bad channels
    bad_chan = flag_rfi_real(spectrum, win, threshold)
    
    ##Zeroing bad channels
    spectrum[bad_chan[:,0]] = 0
    return spectrum




class calibrate_uvh5:

    def __init__(self, datafile, redis_obj):

        #Initializing the pyuvdata object and reading the files
        self.datafile = datafile
        self.uvd = UVData()
        self.uvd.read(datafile, fix_old_proj=False)
        self.metadata = self.get_metadata()
        self.vis_data = self.get_vis_data()
        self.ant_indices = self.get_ant_array_indices()
        self.redis_obj = redis_obj

    def get_metadata(self):
        """
        Reading the metadata from the uvh5 object 
        and adding them to a dictionary
        """
        #make some changes here, each visibiliy could have a different integration time
        extra_keywords = self.uvd.extra_keywords
        
        time_array = np.arange(self.uvd.Ntimes)*self.uvd.integration_time[0]
        metadata  = {'nant_data' : self.uvd.Nants_data,
        'nant_array' : self.uvd.Nants_telescope,
        'ant_names' : list(self.uvd.antenna_names),
        'ant_numbers' : list(self.uvd.antenna_numbers),
        'ant_numbers_data' : self.uvd.get_ants(),
        'nfreqs' : self.uvd.Nfreqs,
        'ntimes' : self.uvd.Ntimes,
        'npols': self.uvd.Npols,
        'nbls' : self.uvd.Nbls,
        'nspws' : self.uvd.Nspws,
        'chan_width': self.uvd.channel_width,
        'intg_time' : self.uvd.integration_time[0],
        'lobs' : self.uvd.Ntimes*self.uvd.integration_time[0],
        'source': self.uvd.object_name.split('.'),
        'telescope' : self.uvd.telescope_name,
        'pol_array' : uvutils.polnum2str(self.uvd.polarization_array),
        'freq_array' : self.uvd.freq_array[0,:],
        'time_array' : time_array,
        'tuning' : extra_keywords['Tuning'],
        'obs_id' : extra_keywords['ObservationID']}
        return metadata

    def get_refant(self):
        observed_antenna_names = [
            self.metadata['ant_names'][self.metadata['ant_numbers'].index(antnum)]
            for antnum in self.metadata['ant_numbers_data']
        ]
        try:
            antdisp = redis_hget_keyvalues(self.redis_obj, "META_antennaDisplacement")
            sorted_antenna_name_list = list(
                dict(
                    sorted(antdisp.items(), key=lambda item: item[1] if item[1] != -1.0 else float('inf'))
                ).keys()
            )
        except:
            sorted_antenna_name_list = observed_antenna_names

        for antname in sorted_antenna_name_list:
            if antname not in self.metadata['ant_names']:
                continue
            
            if antname in BAD_REFANT:
                continue
            antind = self.metadata['ant_names'].index(antname)
            antnum = self.metadata['ant_numbers'][antind]
            if antnum in self.metadata['ant_numbers_data']:
                return antname

        raise RuntimeError(
            f"Cannot select a reference antenna:\n\t"
            f"len(META_antennaDisplacement): {len(antdisp)}\n\t"
            f"BAD_REFANT: {BAD_REFANT}\n\t"
            f"sorted antenna names: {sorted_antenna_name_list}\n\t"
            f"observed antenna names: {observed_antenna_names}\n\t"
        )


    def print_metadata(self):
        #Return string of full observation details

        s = (f" Observations from {self.metadata['telescope']}: \n\
                Source observed: {self.metadata['source']} \n\
                No. of time integrations: {self.metadata['ntimes']} \n\
                Length of time integration: {self.metadata['intg_time']} s \n\
                Length of observations: {self.metadata['lobs']} s \n\
                No. of frequency channels: {self.metadata['nfreqs']} \n\
                Width of frequency channel: {self.metadata['chan_width']/1e+3} kHz\n\
                Start freq: {self.metadata['freq_array'][0]/1e+6} MHz, Stop freq: {self.metadata['freq_array'][-1]/1e+6} MHz \n\
                Observation bandwidth: {(self.metadata['freq_array'][-1] - self.metadata['freq_array'][0])/1e+6} MHz \n\
                No. of spectral windows: {self.metadata['nspws']}  \n\
                Polarization array: {self.metadata['pol_array']} \n\
                No. of polarizations: {self.metadata['npols']}   \n\
                Data array shape: {self.vis_data.shape} \n\
                No. of baselines: {self.metadata['nbls']}  \n\
                No. of antennas present in data: {self.metadata['nant_data']} \n\
                Current antenna list in the data: {self.metadata['ant_numbers_data']} \n\
                No. of antennas in the array: {self.metadata['nant_array']} \n\
                Antenna name: {self.metadata['ant_names']} \n\
                Tuning: {self.metadata['tuning']}\n\
                Observation ID : {self.metadata['obs_id']}")
        return s


    def get_uvw_data(self):
        #Get the UVW info
        uvw_array = self.uvd.uvw_array
        uvw_array = uvw_array.reshape(self.metadata['ntimes'], self.metadata['bls'], 3)
        return uvw_array


    def get_vis_data(self):
        
        """
        Iterate baseline by baseline and collect the
        visibility data
        """
        ant1, ant2 = self.uvd.baseline_to_antnums(self.uvd.baseline_array[:self.metadata['nbls']])
        vis = np.zeros((self.metadata['nbls'], self.metadata['ntimes'], self.metadata['nfreqs'], self.metadata['npols']), dtype = 'complex128')
        for i in range(self.metadata['nbls']):
            vis[i,...] = self.uvd.get_data(ant1[i], ant2[i])

        return np.squeeze(vis)
       
    def write_ms(self, outdir):
        """
        Write the uvh5 data file into measurement set for CASA
        """
        outfile = os.path.join(outdir, os.path.basename(self.datafile).split('.')[0] +'.ms')
        return self.uvd.write_ms(outfile)

    def flag_rfi_vis(self, threshold = 3):
        """
        Flag RFI channels in the visibility data

        """
        print("Starting RFI flagging now")
        t1 = time.time()
        self.vis_data, flagged_visibility_idx = flag_complex_vis_medf(self.vis_data, threshold)
        flagged_freqs = self.derive_flagged_frequencies(flagged_visibility_idx, ref_ant = 'ea21')
        t2 = time.time()
        print(f"Flagging finished in {t2-t1}s")
        return flagged_freqs

    def derive_flagged_frequencies(self, flagged_visibility_idx, ref_ant):
        """
        The output from flagging complex visibilities is the visibility and an n_baselines
        long list of flagged frequency indices for Stokes I.

        Knowing the reference antenna and antenna present in the baselines it is possible to derive a list 
        of flagged frequencies per antenna.
        """
        flagged_frequencies = {}
        for bl in range(len(flagged_visibility_idx)):
            ant0, ant1 = self.ant_indices[bl]
            if ant0 != ant1:
                if len(flagged_visibility_idx[bl]) != 0:
                    if self.metadata['ant_names'][ant1] == ref_ant:
                        flagged_frequencies[self.metadata['ant_names'][ant0]] = self.metadata['freq_array'][np.array(flagged_visibility_idx[bl]).flatten()].tolist()
                    elif self.metadata['ant_names'][ant0] == ref_ant:
                        flagged_frequencies[self.metadata['ant_names'][ant1]] = self.metadata['freq_array'][np.array(flagged_visibility_idx[bl]).flatten()].tolist()
        return flagged_frequencies

    def derive_gains(self, outdir,  ref_ant = 'ea12', flagged_freqs = None, calculate_grade = False):
        """
        Derive gains per antenna/channel/polarizations
        using some of the sdmpy calibration codes

        If calculate_grade is True, apply gains to visibilities and recalculate the gains. From these secondary
        gains, calculate a grade.
        """

        print("Deriving Calibrations now")
        t1 = time.time()
        #Check the ref antenna here, make sure if it is antenna 10.
        antind = int(ref_ant[2:])
        gainsol_dict = gaincal_cpu(self.vis_data, self.metadata['ant_numbers_data'], self.ant_indices,  axis = 0, avg = [1], ref_ant = antind)
        gain = np.squeeze(gainsol_dict['gain_val'])
        gain_ant = gainsol_dict['antennas']

        if calculate_grade:
            gain_grade = calc_gain_grade(gain)
            #we can overwrite the visibility data rather than make a copy as we have already derived our gains and it saves time
            applycal(self.vis_data, gainsol_dict, self.metadata['ant_numbers_data'], self.ant_indices, axis=0, phaseonly=True)
            # self.plot_phases_vs_freq(self.vis_data, os.path.join(outdir,'plots'), plot_amp = True, corrected = True)
            proposed_gainsol_dict = gaincal_cpu(self.vis_data, self.metadata['ant_numbers_data'], self.ant_indices,  axis = 0, avg = [1], ref_ant = antind)
            proposed_gain = np.squeeze(proposed_gainsol_dict['gain_val'])
            proposed_gain_grade = calc_gain_grade(proposed_gain)
            print(f"Calculated proposed gain grade of: {proposed_gain_grade}")
        
        #for i in range(1,29):
        #    ant = "ea"+str(i).zfill(2)
        #    json_gain_dict['ant_gains'][ant] = {}

        json_gain_dict = {'gains':{}, 
                          'freqs_hz': self.metadata['freq_array'].tolist(), 
                          'flagged_hz':flagged_freqs}
        
        #Let's go through each antenna in the gain antenna list and update the gain values
        for i, ant in enumerate(gain_ant):
            ant_str = "ea"+str(ant).zfill(2)
            json_gain_dict['gains'][ant_str] = {}
            json_gain_dict['gains'][ant_str]['gain_pol0_real'] = gain[i, :, 0].real.tolist()
            json_gain_dict['gains'][ant_str]['gain_pol0_imag'] = gain[i, :, 0].imag.tolist()
            json_gain_dict['gains'][ant_str]['gain_pol1_real'] = gain[i, :, 3].real.tolist()
            json_gain_dict['gains'][ant_str]['gain_pol1_imag'] = gain[i, :, 3].imag.tolist()
        json_gain_dict['obs_id'] = self.metadata['obs_id']
        json_gain_dict['ref_ant'] = ref_ant
        if calculate_grade:
            json_gain_dict['grade'] = gain_grade
            json_gain_dict['proposed_gain_grade'] = proposed_gain_grade
        write_out_dict = {}
        write_out_dict[str(min(self.metadata['freq_array'])/1e+6)+","+self.metadata["tuning"]] = json_gain_dict
        #Writting the dictionary as a json file
        outfile_json = os.path.join(outdir, os.path.splitext(os.path.basename(self.datafile))[0] + f"_gain_dict.json")

        try:
            print(f"Writing our the gains per antenna/freq/pols to {outfile_json}")
            with open(outfile_json, "w") as jh:
                json.dump(write_out_dict, jh)
            shutil.chown(outfile_json, "cosmic", "cosmic")
        except:
            print(f"Unable to create file {outfile_json}. Continuing without saving gain dictionary to disk...")
            pass

        t2 = time.time()
        print(f"Took {t2-t1}s for getting solution from {self.metadata['lobs']}s of data")

        print(f"Solution shape: {gainsol_dict['gain_val'].shape}")
        
        return write_out_dict

    def apply_gains(self, gainsol):
        """
        Apply the derived gains to a UVH5 dataset. 
        Also the antenna list in gain has to match with the new dataset, otherwise applying the gains to the
        correlated matrix would be difficult
        """
        data_cp = self.vis_data.copy()
        applycal(data_cp, gainsol, self.metadata['ant_numbers_data'], self.ant_indices, axis=0, phaseonly=False)
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
        
        #check if outdir exists, if not, create dir:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

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
            try:
                print(f"Writing to: {outfile}")
                plt.savefig(outfile, dpi = 150)
            except Exception as e:
                print(f"Encountered an error while saving the plot {e}")
                pass
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
                try:
                    print(f"Writing to: {outfile}")
                    plt.savefig(outfile, dpi = 150)
                except Exception as e:
                    print(f"Encountered an error while saving the plot {e}")
                    pass
                plt.close()
            
    def plot_phases_waterfall(self, data, outdir, track_phase = False):

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
            try:
                plt.savefig(outfile_rr, dpi = 150)
            except:
                pass
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
            try:
                plt.savefig(outfile_ll, dpi = 150)
            except:
                pass
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
                try:
                    plt.savefig(outfile_ph_track, dpi = 150)
                except:
                    pass
                plt.close()   


    def get_res_delays(self, data, outdir, ref_ant = 'ea12'):

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
        outfile_res = os.path.join(outdir, os.path.splitext(os.path.basename(self.datafile))[0]+ f"_res_delay_{tun}.csv")
        try:
            dh = open(outfile_res, "w")
        except:
            print(f"Unable to create {outfile_res}, cannot save delays to file - Aborting run.")
            return None

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
            if ref_ant in ant_base:
                ant_base.remove(ref_ant)
                ant_new = ant_base[0]
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

    def get_phases(self, ref_ant = 'ea12'):

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


    def plot_delays_waterfall(self, data, outdir, track_delay = True):
        
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
            try:
                plt.savefig(outfile_rr, dpi = 150)
            except:
                pass
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
            try:
                plt.savefig(outfile_ll, dpi = 150)
            except:
                pass
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
                try:
                    plt.savefig(outfile_peak, dpi = 150)
                except:
                    pass
                plt.close()    
            
    
    def pub_to_redis(self, phase_out = None, delays_outfile = None, gains_out = None):
        #create channel pubsub object for broadcasting changes to phases/residual-delays
        pubsub = self.redis_obj.pubsub(ignore_subscribe_messages=True)
        if phase_out is not None:
            try:
                pubsub.subscribe("gpu_calibrationphases")
            except redis.RedisError:
                raise redis.RedisError("""Unable to subscribe to gpu_calibrationphases channel to notify of 
                changes to GPU_calibrationPhases changes.""")
            ant_names = phase_out['ant_names']
            phase_vals_0 = phase_out['phases_pol0']
            phase_vals_1 = phase_out['phases_pol1']

            dict_to_pub = {}
            for i, ant in enumerate(ant_names):
                dict_to_pub[ant] = {
                    'freq_array' : phase_out['freqs_hz'],
                    'pol0_phases' : phase_vals_0[i],
                    'pol1_phases' : phase_vals_1[i]
                }
            dict_to_pub['obs_id'] = self.metadata['obs_id']
            self.redis_obj.hset("GPU_calibrationPhases", str(self.metadata['freq_array'][0]/1e+6)+","+self.metadata["tuning"], json.dumps(dict_to_pub))
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
                    'pol0_residual' : residual_delays[i]['res_pol0'],
                    'pol1_residual' : residual_delays[i]['res_pol1']
                }
            dict_to_pub['obs_id'] = self.metadata['obs_id']
            self.redis_obj.hset("GPU_calibrationDelays", str(self.metadata['freq_array'][0]/1e+6)+","+self.metadata["tuning"], json.dumps(dict_to_pub))
            self.redis_obj.publish("gpu_calibrationdelays", json.dumps(True))

        if gains_out is not None:
            try:
                pubsub.subscribe("gpu_calibrationgains")
            except redis.RedisError:
                raise redis.RedisError("""Unable to subscribe to gpu_calibrationdelays channel to notify of 
                changes to GPU_calibrationDelays changes.""")
            redis_publish_dict_to_hash(self.redis_obj, "GPU_calibrationGains", gains_out)
            self.redis_obj.publish("gpu_calibrationgains", json.dumps(True))

def main(uvh5_file_path, args):

    print(f"Processing {uvh5_file_path} now...\n")
    
    out_phase, outfile_delays, out_gains = (None, None, None)

    # Creating an object with the input data file from solutions needed to be derived
    cal_ob = calibrate_uvh5(uvh5_file_path, redis_obj)

    #derive output path
    if args.out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(uvh5_file_path)), "calibration/calibration_gains")
    else:
        out_dir = os.path.join(os.path.abspath(args.out_dir), "calibration/calibration_gains")

    try:
        os.makedirs(out_dir, exist_ok=True)
        save_file_products = True
    except:
        print(f"Unable to create directory {out_dir}, no solutions/metadata from this calibration run will be saved to file.")
        save_file_products = False
        
    #Print the metdata of the input file
    if args.detail:
        detail = cal_ob.print_metadata()
        print(detail)
        if save_file_products:
            with open(os.path.join(out_dir,f'{cal_ob.metadata["obs_id"]}_metadata.txt'), 'w') as f:
                f.write(detail)
    if args.refant is None:
        refant = cal_ob.get_refant()
    else:
        refant = args.refant
    #++++++++++++++++++++++++++++++++++++++++++++++++
    #Use if needed to convert file to a CASA MS format
    #cal_ob.write_ms(args.out_dir)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #Flag the narrowband RFI in the data, use this before calculating delays and gains
    if args.flagrfi:
        flagged_freqs = cal_ob.flag_rfi_vis(threshold = 5)
    else:
        flagged_freqs = None

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Make a bunch of diagnostic plots before applying calibrations
    
    #plot the ampilitude and phase of visibility data
    if args.phasevsfreq:
        cal_ob.plot_phases_vs_freq(cal_ob.vis_data, args.out_dir, plot_amp = True)
    
    #plot the Phase waterfall plots of the visibility
    if args.phasewaterfall:
        cal_ob.plot_phases_waterfall(cal_ob.vis_data, args.out_dir, track_phase = True)

    #plot the Delay waterfall plots of the visibility
    if args.delaywaterfall:
        cal_ob.plot_delays_waterfall(cal_ob.vis_data, args.out_dir, track_delay = True)
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    #Calculate the delays and spit out the delay values per baseline in the out_dir
    if args.gendelay:
        outfile_delays = cal_ob.get_res_delays(cal_ob.vis_data, out_dir, ref_ant = refant)
        shutil.chown(outfile_delays, "cosmic", "cosmic")
       
    if args.genphase:
        antnames, phases = cal_ob.get_phases(ref_ant = refant) # An antenna x time x channel x ?cross-pol?
        out_phase = {
            'ant_names': antnames,
            'freqs_hz': cal_ob.metadata['freq_array'].tolist(),
            'phases_pol0': phases[:,0].tolist(),
            'phases_pol1': phases[:,1].tolist(),
        }
        outfile_phase = os.path.join(out_dir, os.path.splitext(os.path.basename(uvh5_file_path))[0] + '_phasecal.json')
        try:
            with open(outfile_phase, 'w') as fh:
                json.dump(out_phase, fh)
            shutil.chown(outfile_phase, "cosmic", "cosmic")
        except:
            print(f"Unable to create file {outfile_phase}. Continuing without saving phase dictionary to disk...")
            pass

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Derive the gain solutions from the visibility data
    #The gain dictinary obtained from sdmpy
    # Contains the list of antennas, ref antenna used to derive gain and the gain solutions in the form of (nant, ntimes, nfreqs, pols)
    if args.gengain:
        out_gains = cal_ob.derive_gains(out_dir, ref_ant = refant, flagged_freqs = flagged_freqs, calculate_grade=args.calc_gain_grade)
            
    if args.pub_to_redis:
        cal_ob.pub_to_redis(phase_out = out_phase, delays_outfile = outfile_delays, gains_out = out_gains)
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
    
    print(out_dir)
    
if __name__ == '__main__':
    
    # Argument parser taking various arguments
    parser = argparse.ArgumentParser(
        description='Reads UVH5 files, derives delay and gain calibrations, apply to the data, make a bunch of diagnostic plots',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('paths', nargs='*', help = 'UVH5 file/files to derive delay and phase calibrations')
    parser.add_argument('-ad','--apply_dat_file', type = str, required = False, help = 'UVH5 file to apply solutions derived from UVH5 file')
    parser.add_argument('-o','--out_dir', type = str, required = False, help = 'Output directory to save the plots - if not provided, will write out gains to same directory as input uvh5')
    parser.add_argument('--refant', type = str, required = False, help = 'Reference antenna to use in gain derivation')
    parser.add_argument('--flagrfi', action='store_true',
            help = 'If set, flag the narrowband RFI in the dataset')
    parser.add_argument('--gengain', action='store_true',
            help = 'If set, generate a json file of output gain per antenna/freq/pol')
    parser.add_argument('--genphase', action='store_true',
            help = 'If set, generate a file of output phases per antpol')
    parser.add_argument('--calc-gain-grade',action='store_true',
            help = "If set, apply gains to visibilities and calculate a proposed gain grade based on the resultant gains.")
    parser.add_argument('--gendelay', action='store_true',
            help = 'If set, generate a file of output delays per antpol')
    parser.add_argument('--pub-to-redis', action="store_true", help ="Set up a redis object and publish the residual delays and calibration phases to it.")
    parser.add_argument('--detail', action='store_true', help="""
    If specified, will print out and save the UVH5 header to file""")
    parser.add_argument('--phasevsfreq', action='store_true', help="""
    If specified, generate and save plots of phase vs frequency""")
    parser.add_argument('--phasewaterfall', action='store_true', help="""
    If specified, generate and save phase waterfall plots""")
    parser.add_argument('--delaywaterfall', action='store_true', help="""
    If specified, generate and save delay waterfall plots""")
    args = parser.parse_args()

    # try:
    #     # recursive_chown(args.out_dir, "cosmic", "cosmic")
    #     os.system(f"chown cosmic:swdev -R {args.out_dir}")
    # except:
    #     pass

    if len(args.paths) != 0:
        for path in args.paths:
            #iterate through all *.uvh5 files
            if os.path.isfile(path):
                file_path = path
                main(file_path, args)
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.uvh5'):
                            file_path = os.path.join(root, file)
                            main(file_path, args)



