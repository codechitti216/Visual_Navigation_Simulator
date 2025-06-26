#!/bin/bash

# TORCS Data Logger Launcher
# Captures RGB frames, pose, and goal data for ViNT training

echo "=== TORCS ViNT Data Logger ==="
echo "This will launch TORCS and log data to ./vint_torcs_logs/"
echo ""

# Create local data directory
mkdir -p ./vint_torcs_logs

echo "Data will be logged to: ./vint_torcs_logs/{track}_{timestamp}/"
echo ""

# Launch TORCS
echo "Launching TORCS..."
echo "Controls:"
echo "  'l' - Toggle data logging on/off during race"
echo "  'q' - Quit TORCS"
echo ""

# Set environment variables for data logging
export TORCS_DATA_LOGGING=1

# Launch TORCS
torcs

echo ""
echo "TORCS closed. Check ./vint_torcs_logs/ for logged data." 