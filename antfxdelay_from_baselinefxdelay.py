import os
import argparse
import pandas as pd
import datetime
import re

def antfxdelay_from_baselinefxdelay(d_AC : str = "", d_BD : str = "", inpt_fx_delay : str = "", refant=""):
    """
    Parse input tuning CSV files as well as the CSV file of fixed delays presumed to be loaded,
    apply the residual delays from AC and BD files to the loaded fixed delays.
    """
    
    #Parse loaded delays csv file:        
    try:
        inpt_fx_delay = os.path.abspath(inpt_fx_delay)
        fixed_delays = pd.read_csv(inpt_fx_delay, names = ["IF0","IF1","IF2","IF3"],
                            header=None, skiprows=1).to_dict()
    except:
        print(f"Error, unable to parse csv fixed delay file: {inpt_fx_delay}")
        return 0,0
    
    #Current reference antenna is taken to be the zero-field in the input fixed delays:
    if len(refant) == 0:
        #A reference antenna has not been provided, determine previously used one and use that.
        refant = []
        for stream in list(fixed_delays.keys()):
            for ant, val in fixed_delays[stream].items():
                if val == 0:
                    refant += [ant]

        if all(refant):
            refant = refant[0]
        else:
            print(f"""
            Error, loaded fixed delay file appears to contain multiple antenna entries with zero delays. 
            Cannot determine reference antenna.""")
            return 0,0
    else:
        refant = refant

    #Parse tuning 0 delays csv file:
    if len(d_AC) != 0:
        print(f"""
        Received baseline fixed delay file:
        {d_AC} 
        for tuning 0. Modifying loaded fixed delays 
        {inpt_fx_delay}
        with its contents now.
        Using reference antenna {refant}.""")
        tuning_AC_resid = pd.read_csv(os.path.abspath(d_AC), names = ["Baseline","total_pol0","total_pol1","geo","non-geo_pol0","non-geo_pol1","snr_pol0","snr_pol1"],
                                    header=None, skiprows=1).to_dict()
        
        for id, baseline in tuning_AC_resid["Baseline"].items():
            if refant in baseline:
                #We are processing the relavent baseline.
                ant_in_baseline = baseline.split('-')
                #Positionally, if the reference antenna appears in position 2, take the residual to be subtracted from fixed delay
                #otherwise added.
                if ant_in_baseline.index(refant) == 1:
                    #position 2 - apply two polarisations
                    fixed_delays["IF0"][ant_in_baseline[0]] -= tuning_AC_resid["total_pol0"][id]
                    fixed_delays["IF1"][ant_in_baseline[0]] -= tuning_AC_resid["total_pol1"][id]
                elif  ant_in_baseline.index(refant) == 0:
                    #position 2 - apply two polarisations
                    fixed_delays["IF0"][ant_in_baseline[1]] += tuning_AC_resid["total_pol0"][id]
                    fixed_delays["IF1"][ant_in_baseline[1]] += tuning_AC_resid["total_pol1"][id]
                else:
                    print(f"""Error, baseline field for file: 
                    {d_AC}
                    is un-expected. See instance: {baseline}""")
        #If tuning 0 we are following completion of tuning 1, so don't update
        if "_BD" in inpt_fx_delay: 
            filename = "fixed_delay_"+re.sub(' ','',str(datetime.datetime.now()))+'.csv'
        else:
            filename = "fixed_delay_"+re.sub(' ','',str(datetime.datetime.now()))+"_AC"+'.csv'

    #Parse tuning 1 delays csv file:
    if len(d_BD) != 0:
        print(f"""
        Received baseline fixed delay file:
        {d_BD} 
        for tuning 1. Modifying loaded fixed delays
        {inpt_fx_delay}
        with its contents now.
        Using reference antenna {refant}.""")
        tuning_BD_resid = pd.read_csv(os.path.abspath(d_BD), names = ["Baseline","total_pol0","total_pol1","geo","non-geo_pol0","non-geo_pol1","snr_pol0","snr_pol1"],
                                    header=None, skiprows=1).to_dict()
        
        for id, baseline in tuning_BD_resid["Baseline"].items():
            if refant in baseline:
                #We are processing the relavent baseline.
                ant_in_baseline = baseline.split('-')
                #Positionally, if the reference antenna appears in position 2, take the residual to be subtracted from fixed delay
                #otherwise added.
                if ant_in_baseline.index(refant) == 1:
                    #position 2 - apply two polarisations
                    fixed_delays["IF2"][ant_in_baseline[0]] -= tuning_BD_resid["total_pol0"][id]
                    fixed_delays["IF3"][ant_in_baseline[0]] -= tuning_BD_resid["total_pol1"][id]
                elif  ant_in_baseline.index(refant) == 0:
                    #position 2 - apply two polarisations
                    fixed_delays["IF2"][ant_in_baseline[1]] += tuning_BD_resid["total_pol0"][id]
                    fixed_delays["IF3"][ant_in_baseline[1]] += tuning_BD_resid["total_pol1"][id]
                else:
                    print(f"""Error, baseline field for file: 
                    {d_AC}
                    is un-expected. See instance: {baseline}""")

        filename = "fixed_delay_"+re.sub(' ','',str(datetime.datetime.now()))+"_BD"+'.csv'
    pathtosave = os.path.join(os.path.dirname(inpt_fx_delay),filename)
    print(f"Saving new fixed delays to: {pathtosave}")
    fixed_delays = pd.DataFrame.from_dict(fixed_delays)
    fixed_delays.to_csv(pathtosave)
    if os.path.isfile(pathtosave):
        return pathtosave
    else:
        print(f"Saved file {pathtosave} does not exist.")
        return 0,0

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description=("""Accept 2 input csv's (one for each tuning) containing baseline to delay mappings, apply residuals to the
    current applied delays csv and save it to the same location.""")
    )
    parser.add_argument('--d-AC', type=str, default="", required=False, help="First tuning set of delay values to apply.")
    parser.add_argument('--d-BD', type=str, default="", required=False, help="Second tuning set of delay values to apply.")
    parser.add_argument("-f","--fixed-delay-to-update", type=str, required=True, help="""
    csv file path to latest fixed delays that must be modified by the residual delays calculated in this script.
    Reference antenna is maintained to be the same.""")

    args = parser.parse_args()
    antfxdelay_from_baselinefxdelay(args.d_AC, args.d_BD, inpt_fx_delay=args.fixed_delay_to_update)