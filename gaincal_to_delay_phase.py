"""
Read a gaincal JSON file and output a per-antenna delay and phase calibration.
"""
import os
import argparse
import numpy as np
import json

NANTS=16

def main(args):
    assert os.path.exists(args.infile)
    # load multiple files if necessary, building up a list of freqs and phases
    # which need not be contiguous
    freqs_hz = []
    phases_pol0 = {}
    phases_pol1 = {}
    with open(args.infile, 'r') as fh:
        indat = json.load(fh)
    freqs_hz += indat['freqs_hz'] 
    print(len(indat['phases_pol0']))
    print(len(phases_pol0))
    ant_names = indat['ant_names']
    for ant in range(NANTS):
        ant_name = ant_names[ant]
        if not ant_name in phases_pol0:
            phases_pol0[ant_name] = []
        if not ant_name in phases_pol1:
            phases_pol1[ant_name] = []
        phases_pol0[ant_name] += indat['phases_pol0'][ant]
        phases_pol1[ant_name] += indat['phases_pol1'][ant]
    # 

    freqs_hz = np.array(freqs_hz)
    #phases_pol0 = np.array(phases_pol0)
    #phases_pol1 = np.array(phases_pol1)

    for ant_name in phases_pol0.keys():
        print('Antenna %s' % ant_name)
        #unwrap_phases
        phases0 = np.array(np.unwrap(phases_pol0[ant_name]))
        phases1 = np.array(np.unwrap(phases_pol1[ant_name]))
        # Fit delay
        pol0_phase_slope, pol0_phase_intercept = np.polyfit(freqs_hz, phases0, 1)
        pol1_phase_slope, pol1_phase_intercept = np.polyfit(freqs_hz, phases1, 1)
        residual0 = phases0 - (pol0_phase_slope*freqs_hz)
        residual1 = phases1 - (pol1_phase_slope*freqs_hz)
        residual0 = residual0 % (2*np.pi)
        residual1 = residual1 % (2*np.pi)
        print('Pol 0 delay ns: %.2f' % (pol0_phase_slope/(2*np.pi) * 1e9))
        #print('Pol 0 phase:')
        #print(residual0)
        print('Pol 1 delay ns: %.2f' % (pol1_phase_slope/(2*np.pi) * 1e9))
        #print('Pol 1 phase:')
        #print(residual1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Read a gaincal JSON file and output a per-antenna delay and phase calibration.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i','--infile', type = str, required = True, help = 'input gaincal JSON file')
    parser.add_argument('-o','--out_dir', type = str, required = True, help = 'Output directory for calibration JSON')
    args = parser.parse_args()
    main(args)



