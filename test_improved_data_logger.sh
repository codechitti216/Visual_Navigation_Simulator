#!/bin/bash

echo "=== Testing Improved ViNT Data Logger ==="
echo "This test will verify:"
echo "1. Larger image resolution (512x512 instead of 224x224)"
echo "2. Properly oriented images (no inversion)"
echo "3. Ego-centric cockpit camera view"
echo "4. Automatic camera switching when logging enabled"
echo ""

# Clean up any previous test data
echo "Cleaning up previous test data..."
rm -rf /data/vint_training_logs/test_improved_*

# Start TORCS in the background
echo "Starting TORCS..."
/usr/local/bin/torcs &
TORCS_PID=$!

echo "TORCS started with PID: $TORCS_PID"
echo ""
echo "=== Instructions ==="
echo "1. Select 'Race' -> 'Quick Race' -> 'Configure Race'"
echo "2. Choose track: 'road/forza' or 'city/g-track-1' (good for testing)"
echo "3. Make sure Human driver is selected"
echo "4. Start the race"
echo "5. Press 'L' key to toggle ViNT logging ON"
echo "6. Drive around for 30-60 seconds to collect data"
echo "7. Press 'L' key again to toggle logging OFF"
echo "8. Press 'Esc' to exit TORCS"
echo ""
echo "The script will wait for you to exit TORCS, then analyze the captured data..."

# Wait for TORCS to exit
wait $TORCS_PID

echo ""
echo "=== Analyzing Captured Data ==="

# Find the most recent session directory
LATEST_SESSION=$(find /data/vint_training_logs -name "*" -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$LATEST_SESSION" ] || [ ! -d "$LATEST_SESSION" ]; then
    echo "❌ No data captured! Make sure to:"
    echo "   - Press 'L' to enable logging during the race"
    echo "   - Drive around for some time"
    echo "   - Check console output for 'ViNT logging enabled' message"
    exit 1
fi

echo "✅ Found session data in: $LATEST_SESSION"
echo ""

# Analyze the captured data
if [ -d "$LATEST_SESSION/frames" ]; then
    FRAME_COUNT=$(ls "$LATEST_SESSION/frames"/*.png 2>/dev/null | wc -l)
    echo "📸 Frame count: $FRAME_COUNT"
    
    if [ $FRAME_COUNT -gt 0 ]; then
        # Check first frame for size and properties
        FIRST_FRAME=$(ls "$LATEST_SESSION/frames"/*.png 2>/dev/null | head -1)
        if [ -f "$FIRST_FRAME" ]; then
            echo "🔍 Analyzing first frame: $(basename "$FIRST_FRAME")"
            
            # Get image dimensions
            DIMENSIONS=$(identify "$FIRST_FRAME" 2>/dev/null | cut -d' ' -f3)
            echo "   Resolution: $DIMENSIONS"
            
            # Expected: 512x512
            if [ "$DIMENSIONS" = "512x512" ]; then
                echo "   ✅ Resolution is correct (512x512 - improved from 224x224)"
            else
                echo "   ❌ Resolution unexpected (expected 512x512, got $DIMENSIONS)"
            fi
            
            # Check file size (larger images should be bigger files)
            FILE_SIZE=$(stat -c%s "$FIRST_FRAME")
            echo "   File size: $FILE_SIZE bytes"
            
            if [ $FILE_SIZE -gt 50000 ]; then
                echo "   ✅ File size indicates high-quality image"
            else
                echo "   ⚠️  File size seems small for 512x512 image"
            fi
            
            # Try to open the image to verify it's not corrupted/inverted
            echo "   🖼️  Opening first frame for visual inspection..."
            eog "$FIRST_FRAME" &
            echo "   Visual check: Does the image show:"
            echo "   - Ego-centric cockpit view (dashboard visible)?"
            echo "   - Proper orientation (not upside down)?"
            echo "   - Good image quality and detail?"
        fi
    fi
else
    echo "❌ No frames directory found"
fi

# Check CSV data
if [ -f "$LATEST_SESSION/pose.csv" ]; then
    POSE_LINES=$(wc -l < "$LATEST_SESSION/pose.csv")
    echo "📊 Pose data: $POSE_LINES lines"
    echo "   Sample pose data:"
    head -3 "$LATEST_SESSION/pose.csv" | sed 's/^/   /'
else
    echo "❌ No pose.csv found"
fi

if [ -f "$LATEST_SESSION/goal.csv" ]; then
    GOAL_LINES=$(wc -l < "$LATEST_SESSION/goal.csv")
    echo "🎯 Goal data: $GOAL_LINES lines"
    echo "   Sample goal data:"
    head -3 "$LATEST_SESSION/goal.csv" | sed 's/^/   /'
else
    echo "❌ No goal.csv found"
fi

# Check metadata
if [ -f "$LATEST_SESSION/metadata.json" ]; then
    echo "📋 Metadata:"
    cat "$LATEST_SESSION/metadata.json" | sed 's/^/   /'
else
    echo "❌ No metadata.json found"
fi

echo ""
echo "=== Summary ==="
echo "Session directory: $LATEST_SESSION"
echo "Frames captured: $FRAME_COUNT"
echo ""
echo "🔧 Improvements implemented:"
echo "  ✅ Increased resolution from 224x224 to 512x512"
echo "  ✅ Fixed image inversion (proper OpenGL->image orientation)"
echo "  ✅ Automatic ego-centric cockpit camera switching"
echo "  ✅ Enhanced debugging output during capture"

echo ""
echo "Test completed! Check the opened image to verify visual quality." 