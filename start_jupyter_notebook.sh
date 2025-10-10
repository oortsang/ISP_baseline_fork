#!/bin/bash

my_ip_addr=$(hostname -i)
port_num=$((1024+RANDOM))
echo "Connect to ${my_ip_addr}:${port_num}"
echo "Copy/paste:"
echo "ssh -NL 8898:${my_ip_addr}:${port_num} ttic-cluster"
# echo "ssh -NL 8898:${my_ip_addr}:${port_num} beehive"
echo ""
echo ""
sleep 1
jupyter-notebook --no-browser --port=${port_num} --ip=${my_ip_addr}
