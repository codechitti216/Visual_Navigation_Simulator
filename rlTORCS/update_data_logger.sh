#!/bin/bash

echo "=== Updating TORCS Data Logger ==="

cd torcs-1.3.6/src/modules/graphic/ssggraph

# Backup current system module
sudo cp /usr/local/lib/torcs/modules/graphic/ssggraph.so /usr/local/lib/torcs/modules/graphic/ssggraph.so.backup

echo "Building other required object files..."

# Compile other required files if they don't exist
for file in grmain grboard ssggraph grcam grcar grscreen grscene grutil grshadow grsmoke grskidmarks grloadac grmultitexstate grvtxtable grtrackmap grtexture grcarlight CarSoundData TorcsSound PlibSoundInterface OpenalSoundInterface grsound SoundInterface; do
    if [ ! -f "$file.o" ]; then
        echo "Compiling $file.cpp..."
        g++ -I/home/surya/Desktop/Surya/rlTORCS/torcs-1.3.6/export/include -I/home/surya/Desktop/Surya/rlTORCS/torcs-1.3.6 -I/usr/include/opencv4 -g -O2 -Wall -fPIC -fno-strict-aliasing -O2 -DUSE_RANDR_EXT -DGL_GLEXT_PROTOTYPES -D_SVID_SOURCE -D_BSD_SOURCE -DSHM -DHAVE_CONFIG_H -c ${file}.cpp
    fi
done

echo "Linking updated graphics module..."
g++ -shared -o ssggraph.so *.o -L/home/surya/Desktop/Surya/rlTORCS/torcs-1.3.6/export/lib -lplibsl -lplibsm -lplibsg -lplibssg -lplibul -lplibssgaux -lopenal -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

if [ -f "ssggraph.so" ]; then
    echo "Installing updated module..."
    sudo cp ssggraph.so /usr/local/lib/torcs/modules/graphic/ssggraph.so
    echo "✅ Data logger updated successfully!"
    echo ""
    echo "Now run TORCS and press 'l' to test data logging with debug output."
else
    echo "❌ Build failed"
    exit 1
fi 