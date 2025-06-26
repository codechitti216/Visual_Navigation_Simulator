#!/bin/bash

# Set up environment for running TORCS from the compiled version
export LD_LIBRARY_PATH="/home/surya/Desktop/Surya/rlTORCS/Visual_Navigation_Simulator/torcs-1.3.6/export/lib:$LD_LIBRARY_PATH"

# Change to the TORCS directory so it can find data files
cd /home/surya/Desktop/Surya/rlTORCS/Visual_Navigation_Simulator/torcs-1.3.6

echo "Starting TORCS with ViNT logging enabled..."
echo "Logs will be saved to: /home/surya/Desktop/Surya/rlTORCS/Visual_Navigation_Simulator/vint_torcs_logs/"
echo "Press F2/F3/F4 to test camera switching and see debug output"
echo "Library path: $LD_LIBRARY_PATH"
echo ""

# Run TORCS
./src/linux/torcs-bin 