from cosmic.redis_actions import redis_obj, redis_hget_keyvalues
import redis
from cosmic.fengines import ant_remotefeng_map
import time
import numpy as np
import logging
import json
from logging.handlers import RotatingFileHandler
from influxdb_client_3 import InfluxDBClient3, Point  # Updated to v3
import os
import argparse

LOGFILENAME = "/home/cosmic/logs/Calibration.log"

SERVICE_NAME = os.path.splitext(os.path.basename(__file__))[0]

logger = logging.getLogger('calibration_logger')
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fh = RotatingFileHandler(LOGFILENAME, mode = 'a', maxBytes = 512, backupCount = 0, encoding = None, delay = False)
fh.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter("[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s")

# add formatter to ch
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.addHandler(fh)

delay_update_channel = "update_calibration_delays"
phase_update_channel = "update_calibration_phases"

def calibration_logger(influxdb_token):
    # InfluxDB v3 Setup
    database = "delay_influxdb" # Pushing to the same DB as delay_logger
    org = "seti"

    client = InfluxDBClient3(host='http://localhost:8181', token=influxdb_token, org=org, database=database)

    logger.info("Starting Calibration logger...\n")

    # redis channel listening:
    pubsub = redis_obj.pubsub(ignore_subscribe_messages=True)
    for channel in [delay_update_channel, phase_update_channel]:
        try:
            pubsub.subscribe(channel) 
        except redis.RedisError:
            logger.error(f'Subscription to `{channel}` unsuccessful.')
    
    for message in pubsub.listen():
        try:
            json_message = json.loads(message.get('data'))
        except:
            logger.warning(f"Unable to json parse the triggered channel data. Continuing...")
            continue

        if message['channel'] == delay_update_channel:
            # Fixed delays logging
            if json_message:
                time_now = int(time.time_ns())
                points_to_write = []
                
                loaded_delay_file = redis_hget_keyvalues(redis_obj, "CAL_fixedValuePaths", "fixed_delay")
                pt_delay_path = Point("fix_paths").field("fixed_delay_path", str(loaded_delay_file["fixed_delay"])).time(time_now)
                points_to_write.append(pt_delay_path)
                
                fixed_delays = redis_hget_keyvalues(redis_obj, "META_calibrationDelays")
                for ant, delays in fixed_delays.items():
                    ant_calib_delays = np.fromiter(delays.values(),dtype=float)
                    for stream in range(4):
                        # Load fixed delays contents
                        pt_fixed_delay = Point("fix_delays").tag("ant",ant).tag("stream", str(stream)).field("fixed_delay_ns", float(ant_calib_delays[stream])).time(time_now)
                        points_to_write.append(pt_fixed_delay)
                        
                # Inside calibration_logger.py
                if points_to_write:
                    print(f"Writing {len(points_to_write)} points to {database}...")
                    client.write(record=points_to_write)
                    print("Write successful.")

        if message['channel'] == phase_update_channel:
            if json_message:
                time_now = int(time.time_ns())
                points_to_write = []
                
                calibration_phase_grade = redis_hget_keyvalues(redis_obj, "CAL_fixedValuePaths", ["fixed_phase","grade"])
                
                pt_phase_path = Point("fix_paths").field("fixed_phase_path", str(calibration_phase_grade["fixed_phase"])).time(time_now)
                points_to_write.append(pt_phase_path)
                
                pt_grade = Point("fix_paths").field("calibration_grade", float(calibration_phase_grade["grade"])).time(time_now)
                points_to_write.append(pt_grade)
                
                ant_feng_map = ant_remotefeng_map.get_antennaFengineDict(redis_obj)
                ant_phase_cal_map = redis_hget_keyvalues(redis_obj, "META_calibrationPhases")   
                
                for ant, cal_phase in ant_phase_cal_map.items():
                    status = -1  # Default to "Unknown/Unreachable"
                    
                    try:
                        if ant not in ant_feng_map:
                            logger.warning(f"{ant} present in Redis phases but not in F-Engine map. Skipping...")
                        else:
                            feng = ant_feng_map[ant]
                            cal_phase_correct = []
                            expected_cal_phase = (np.array(cal_phase, dtype=float) + np.pi) % (2 * np.pi) - np.pi
                            
                            for stream in range(expected_cal_phase.shape[0]):
                                hardware_phase = np.array(feng.phaserotate.get_phase_cal(stream), dtype=float)
                                cal_phase_correct += [bool(np.all(np.isclose(expected_cal_phase[stream, :],
                                                              hardware_phase, atol=1e-1)))] 
                            
                            # Calculate final Boolean status (1 = Correct, 0 = Mismatch)
                            status = int(all(cal_phase_correct))
                            
                    except Exception as e:
                        logger.error(f"Failed to verify phase calibration for {ant}: {e}")
                        status = -1 

                    pt_phase_correct = Point("delay_state").tag("ant", ant) \
                        .field("phase_cal_correct", status) \
                        .time(time_now)
                    points_to_write.append(pt_phase_correct)
                    
                # Inside calibration_logger.py
                if points_to_write:
                    print(f"Writing {len(points_to_write)} points to {database}...")
                    client.write(record=points_to_write)
                    print("Write successful.")

def cli_calibration_logger():
    parser = argparse.ArgumentParser(
    description=("Set up the Calibration logger.")
    )
    parser.add_argument(
    "-c", "--clean", action="store_true",help="Delete the existing log file and start afresh.",
    )
    args = parser.parse_args()
    if os.path.exists(LOGFILENAME) and args.clean:
        print("Removing previous log file...")
        os.remove(LOGFILENAME)
    else:
        print("Nothing to clean, continuing...")

    if "INFLUXDB_TOKEN" in os.environ:
        influxdb_token = os.environ["INFLUXDB_TOKEN"]

    calibration_logger(influxdb_token)

if __name__ == "__main__":
    cli_calibration_logger()