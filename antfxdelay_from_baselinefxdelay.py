#Get the delay from the .csv files and track them as a function of time
import csv
import glob
import sys, os
from matplotlib import pyplot as plt
import numpy as np
import numpy.ma as ma

data_path1 = "/home/svarghes/benchmark_test/rawdata/B_config/AC_delay/delays_TCOS0001*"
data_path2 = "/home/svarghes/benchmark_test/rawdata/B_config/BD_delay/delays_TCOS0001*"

files1 = sorted(glob.glob(data_path1))
files2 = sorted(glob.glob(data_path2))

#Dictionary to store all the collected values
#Dictionary of the order
# main_dict = {'ant':{'mjd_time':{ 'AC':{ 'delay_pol0: '', 'delay_pol1':'' }, BD :{'delay_pol0' : '', 'delay_pol1' : ''}}}

data_dict = {}
avg_dict = {}
for i in range(1,29):
    ant = "ea"+str(i).zfill(2)
    data_dict[ant] = {}
    avg_dict[ant] = {}

print(data_dict)

#Reference antenna
ref = 'ea10'

for f,filename in enumerate(files1):
    print(f"Getting AC values from {filename}")
    basename_split1 = os.path.basename(filename).split('AC')[0]
    mjd_time = float(basename_split1.split('_')[3].split('.')[1] + basename_split1.split('_')[3].split('.')[2]) 
    #source = basename_split1[5]
    #freq = float(basename_split1[7].split['-'][0])
    with open(filename, mode='r') as csv_file1: 
        csv_reader1 = csv.DictReader(csv_file1)
        for line in csv_reader1:
            bls = line['Baseline']
            ng0 = float(line['non-geo_pol0'])
            ng1 = float(line['non-geo_pol1'])
            
            #using total_pol since the delay tracking was enabled. Turn it off otherwise
            #ng0 = float(line['total_pol0'])
            #ng1 = float(line['total_pol1'])

            if line['snr_pol0'] != '+00000000nan' and line['snr_pol1'] != '+00000000nan':
                snr0 = float(line['snr_pol0'])
                snr1 = float(line['snr_pol1'])
            else:
                continue

            ants_base = bls.split('-')
    
            if ref in ants_base and (snr0 > 4.0) and (snr1 > 4.0):
                #print(ants_base)
                #print(ref)
                if ants_base[0] ==  ref:
                    ng0 = -ng0
                    ng1 = -ng1
                ants_base.remove(ref)
                ant_new = ants_base[0]
                #print(ant_new)  
                #print(line)  
                data_dict[ant_new][mjd_time] = {'AC':{ 'pol0': ng0, 'pol1': ng1}}    
                #print (ant_new, mjd_time, data_dict[ant_new][mjd_time])

    basename_split2 =  os.path.basename(files2[f]).split('BD')[0]
    if basename_split1 == basename_split2:
        print(f"Getting BD values from {files2[f]}")

        with open(files2[f], mode='r') as csv_file2: 
            csv_reader2 = csv.DictReader(csv_file2)
            for line in csv_reader2:
                #print(line['Baseline'], line['non-geo_pol0'])
                bls = line['Baseline']
                ng0 = float(line['non-geo_pol0'])
                ng1 = float(line['non-geo_pol1'])
                snr0 = float(line['snr_pol0'])
                snr1 = float(line['snr_pol1'])
                ants_base = bls.split('-')
                
        
                if ref in ants_base and snr0 > 4.0 and snr1 > 4.0:
                    if ants_base[0] ==  ref:
                        ng0 = -ng0
                        ng1 = -ng1
                    ants_base.remove(ref)
                    ant_new = ants_base[0]  
                    #print(line)
                    #print (ant_new, mjd_time, data_dict[ant_new][mjd_time])
                    if mjd_time in data_dict[ant_new]:
                        data_dict[ant_new][mjd_time]['BD'] = {'pol0': ng0, 'pol1': ng1}
    else:
        print("No matching files to get the BD values") 

for key in data_dict.keys():
    print(key, data_dict[key])
r = np.random.rand(28)
g = np.random.rand(28)
b = np.random.rand(28)
clr = np.round(np.linspace(0.2,0.8, 28),2)

#Writing averaged values into a csv file
dh = open(f"latest_delays_B_config.csv", "w")
dh.write(",".join(
            [
                "antenna",
                "IF0",
                "IF1",
                "IF3",
                "IF4"
            ]
        )+"\n")

print("Plotting now")
#Plotting the delay values from the dictionary
fig, axs = plt.subplots(5, 5, sharex  = True, sharey = False, constrained_layout=True, figsize = (12,12))
row = 0
col = 0
for i,key in enumerate(data_dict.keys()):
    data_ant = data_dict[key]
    if data_ant:
        times = data_ant.keys()
        times_ar = np.zeros(len(times))
        vals_ar = np.zeros((len(times),4)) # Storing the values into an array
        vals = ma.MaskedArray(vals_ar)
        print(key)
        for i, time in enumerate(times):
            times_ar[i] = time
            #try:
            ac0 = data_ant[time]['AC']['pol0']
            vals[i,0] = ac0
            #except KeyError:
            #    vals[i,0] = ma.masked
            
            #try:
            ac1 = data_ant[time]['AC']['pol1']
            vals[i,1] = ac1
            #except KeyError:
            #vals[i,1] = ma.masked
            
            #try:
            bd0 = data_ant[time]['BD']['pol0']
            vals[i,2] = bd0
            #except KeyError:
            #    vals[i,2] = ma.masked

            #try:
            bd1 = data_ant[time]['BD']['pol1']
            vals[i,3] = bd1
            #except KeyError:
            #    vals[i,3] = ma.masked

            #vals[i,:] = [ac0, ac1, bd0, bd1]

        ac0_avg, ac1_avg, bd0_avg, bd1_avg   = np.mean(vals,axis=0)
        print( key, ac0_avg, ac1_avg, bd0_avg, bd1_avg)
        #Writing the values as negative to match the FPGA convention
        dh.write(f"{key},{-ac0_avg:+012.03f},{-ac1_avg:+012.03f},{-bd0_avg:+012.03f},{-bd1_avg:+012.03f}\n")
        min = np.min(vals)
        max = np.max(vals)
        
        #AC
        #axs[row, col].plot(np.round(times_ar,2), vals[:,0], '.', color = 'b', markerfacecolor = 'none', label = '0') #plotting single pol values
        #axs[row, col].plot(np.round(times_ar,2), vals[:,1], '.', color = 'r', markerfacecolor = 'none', label = '1') #plotting single pol values
        
        #BD
        axs[row, col].plot(np.round(times_ar,2), vals[:,2], '.', color = 'b', markerfacecolor = 'none', label = '0') #plotting single pol values
        axs[row, col].plot(np.round(times_ar,2), vals[:,3], '.', color = 'r', markerfacecolor = 'none', label = '1') #plotting single pol values
        
        
        axs[row, col].set_ylim(min-50, max+300)
        axs[row, col].legend(loc='upper right')
        axs[row, col].set_title(f"antenna: {key}")
        col += 1
        if col == 5:
           row += 1
           col = 0 
        #plt.plot(time, val1, 's', color = clr[i], label = key)
    else:
        dh.write(f"{key},{0:+012.03f},{0:+012.03f},{0:+012.03f},{0:+012.03f}\n")    
dh.close()
fig.supxlabel("MJD time")
fig.supylabel("Non-geometrical delay(ns)")
#fig.suptitle("Non-geometrical delays per antenna vs time, IF:AC")
fig.suptitle("Non-geometrical delays per antenna vs time, IF:BD")
#plt.legend(loc='upper right')
plt.show()
   
