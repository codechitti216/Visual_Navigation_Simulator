#!/bin/bash

# Set up library path for compiled TORCS
export LD_LIBRARY_PATH=./export/lib:$LD_LIBRARY_PATH

# Create symbolic links for data if they don't exist
if [ ! -L data/fonts ]; then
    ln -sf data/fonts data/fonts
fi
if [ ! -L data/img ]; then
    ln -sf data/img data/img
fi

echo "Starting TORCS with clean camera debugging..."
echo "When you press F2-F11, you'll see clean output like:"
echo "  Camera Debug: curCamHead = X, Camera ID = Y"
echo "  Camera Change: curCamHead changing from X to Y"
echo ""
echo "Note the curCamHead value for your preferred camera view!"
echo ""

# Run the compiled TORCS binary
./src/linux/torcs-bin 