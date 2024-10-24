#!/bin/bash


sudo apt-get update
sudo apt-get install python3 python3-pip python3.10-venv -y
python3 -m venv dst_env
source dst_env/bin/activate
pip3 install torch torchvision
