#!/bin/bash 

if [[ $EUID > 0 ]]
then 
  echo "Please run with super-user privileges"
  exit 1
else
	cp ./calibrationlogger.service /etc/systemd/system/
	systemctl disable calibrationlogger.service
	systemctl daemon-reload
	systemctl enable calibrationlogger.service
fi