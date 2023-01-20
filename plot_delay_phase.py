import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse



def plot_delay_phase(args):
    """
    Function to collect the delay and phase information
    from the residual files generated after the calibration step
    and plot them
    """
    delay_file = args.delay_file
    phase_file = args.phase_file

    #opening each json files
    with open(delay_file) as dh:
        delay_dict = json.load(dh)

    with open(phase_file) as ph:
        phase_dict = json.load(ph)

    #print(delay_dict.keys())
    #print(phase_dict.keys())
   
    #Getting the antenna info from the keys
    antennas = list(delay_dict.keys())

    if len(antennas) > 0:
        test_delay = delay_dict[antennas[0]]
        test_phase = phase_dict[antennas[0]]
    else:
        sys.exit("No antenna key in the files")

    delay_shape = tuple([len(antennas),])+np.squeeze(test_delay).shape
    phase_shape = tuple([len(antennas),])+np.squeeze(test_phase).shape
    
    #Saving the delay and phase value into an array for easy plotting
    delay_dat = np.zeros(delay_shape)
    phase_dat = np.zeros(phase_shape)

    # Grabing the values
    for i, ant in enumerate(antennas):
        delay_dat[i,:] = delay_dict[ant]
        phase_dat[i,...] = phase_dict[ant]
    
    #covering the delay to ns
    delay_dat *= 1e+9
    
    #plotting the residual delays vs antennas

    fig, ax = plt.subplots(constrained_layout=True, figsize = (10,6))

    ax.plot(delay_dat[:,0], '.',  label = "AC0")
    ax.plot(delay_dat[:,1], '.',  label = "AC1")
    ax.plot(delay_dat[:,2], '.',  label = "BD0")
    ax.plot(delay_dat[:,3], '.',  label = "BD1")

    ax.set_title("Residual Delays vs Antennas")
    ax.legend(loc = 'upper right')
    ax.set_xticks(np.arange(len(antennas)))
    ax.set_xticklabels(antennas)
    
    fig.supylabel("Residual Delays (s)")
    fig.supxlabel("Antennas")
    plt.savefig("delays_vs_ant.png", dpi = 150)
    plt.show()
    plt.close()
    

    #make a grid plot of phase vs freq for all antennas
    grid_x = 6
    grid_y = 5
    
    fig, axs = plt.subplots(grid_x, grid_y, sharex  = True, sharey = True, constrained_layout=True, figsize = (12,14))
    
    for i in range(grid_x):
        for j in range(grid_y):
            ant_ind = (i*grid_y)+j
            if ant_ind < len(antennas):
                
                axs[i,j].plot(phase_dat[ant_ind,0,:], '.',  label = "AC0")
                axs[i,j].plot(phase_dat[ant_ind,1,:], '.',  label = "AC1")
                axs[i,j].plot(phase_dat[ant_ind,2,:], '.',  label = "BD0")
                axs[i,j].plot(phase_dat[ant_ind,3,:], '.',  label = "BD1")
                axs[i,j].set_title(f"{antennas[ant_ind]}")
                axs[i,j].legend(loc = 'upper right')
            
    fig.suptitle("Residual Phase vs Freq ")
    fig.supylabel("Residual Phase (degrees)")
    fig.supxlabel("Frequency Channels")
    plt.savefig("phase_vs_antennas.png", dpi = 150)
    plt.show()
    plt.close() 

if __name__ == '__main__':
    
    # Argument parser taking various arguments
    parser = argparse.ArgumentParser(
        description='Reads delay and phase JSON file to make plots',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--delay_file', type = str, required = True, help = 'JSON Delay file')
    parser.add_argument('-p','--phase_file', type = str, required = True, help = 'JSON Phase file')
    parser.add_argument('-o','--out_dir', type = str, required = False, default = '.', help = 'Output directory to save the plots')
    args = parser.parse_args()
    plot_delay_phase(args)
