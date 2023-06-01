from cosmic.redis_actions import redis_obj, redis_hget_keyvalues
import redis
from cosmic.fengines import ant_remotefeng_map
import time
import numpy as np
import logging
import json
from logging.handlers import RotatingFileHandler
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
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
    #influxdb stuff:
    bucket = "delays"
    org="seti"
    client = InfluxDBClient(url='http://localhost:8086', token=influxdb_token)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    logger.info("Starting Calibration logger...\n")

    #redis channel listening:
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
            #Fixed delays logging
            if json_message:
                time_now = time.time_ns()
                loaded_delay_file = redis_hget_keyvalues(redis_obj, "CAL_fixedValuePaths", "fixed_delay")
                value = Point("fix_paths").field("fixed_delay_path",loaded_delay_file["fixed_delay"]).time(time_now)
                write_api.write(bucket, org, value)   
                fixed_delays = redis_hget_keyvalues(redis_obj, "META_calibrationDelays")
                for ant, delays in fixed_delays.items():
                    ant_calib_delays = np.fromiter(delays.values(),dtype=float)
                    for stream in range(4):
                        #Load fixed delays contents
                        value = Point("fix_delays").tag("ant",ant).tag("stream",stream).field("fixed_delay_ns",ant_calib_delays[stream]).time(time_now)
                        write_api.write(bucket, org, value)

        if message['channel'] == phase_update_channel:
            if json_message:
                time_now = time.time_ns()
                #arbitraly check grade in this case
                calibration_phase_grade = redis_hget_keyvalues(redis_obj, "CAL_fixedValuePaths", ["fixed_phase","grade"])
                value = Point("fix_paths").field("fixed_phase_path",calibration_phase_grade["fixed_phase"]).time(time_now)
                write_api.write(bucket, org, value)
                value = Point("fix_paths").field("calibration_grade",calibration_phase_grade["grade"]).time(time_now)
                write_api.write(bucket, org, value)
                ant_feng_map = ant_remotefeng_map.get_antennaFengineDict(redis_obj)
                ant_phase_cal_map = redis_hget_keyvalues(redis_obj, "META_calibrationPhases")   
                for ant, cal_phase in ant_phase_cal_map.items():
                    cal_phase_correct = []
                    feng = ant_feng_map[ant]
                    expected_cal_phase = (np.array(cal_phase,dtype=float) + np.pi) % (2 * np.pi) - np.pi
                    for stream in range(expected_cal_phase.shape[0]):
                        cal_phase_correct += [bool(np.all(np.isclose(expected_cal_phase[stream,:],
                                        np.array(feng.phaserotate.get_phase_cal(stream),dtype=float), atol=1e-1)))] 
                    value = Point("delay_state").tag("ant",ant).field("phase_cal_correct",int(all(cal_phase_correct))).time(time_now)
                    write_api.write(bucket, org, value)

if __name__ == "__main__":
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