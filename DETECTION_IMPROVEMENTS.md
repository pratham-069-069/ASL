# Sign Language Detection Improvements

## Overview
The main.py file has been enhanced to reduce false positives and improve detection accuracy during sign language translation.

## New Features

### 1. Detection Box 📦
- **What it does**: Only processes hand signs when your hand is inside a designated zone on screen
- **Visual feedback**: 
  - **RED box** = Hand not detected in zone
  - **GREEN box** = Hand detected, processing signs
- **Benefit**: Prevents false detections from random hand movements outside the zone

### 2. Stability Check ✅
- **What it does**: Requires 3 consecutive frames with the same prediction before accepting it
- **Benefit**: Eliminates false positives during transitions between different signs
- **How it works**: Prediction buffer stores recent predictions and only confirms when they're consistent

### 3. Higher Confidence Threshold 🎯
- **What it does**: Only accepts predictions with 92% or higher confidence (increased from 90%)
- **Benefit**: More strict filtering reduces incorrect detections
- **Display**: Shows confidence level on screen in real-time

### 4. Cooldown System ⏱️
- **What it does**: Prevents the same letter from being detected multiple times rapidly
- **Default**: 15 frames (~0.5 seconds) cooldown between detections
- **Benefit**: Avoids repeated detection of the same sign

## Adjustable Parameters

You can fine-tune these values at the top of `main.py` (lines 29-42):

### Detection Box Size
```python
BOX_LEFT = 0.25      # 25% from left (0.0-1.0)
BOX_RIGHT = 0.75     # 75% from left (0.0-1.0)
BOX_TOP = 0.15       # 15% from top (0.0-1.0)
BOX_BOTTOM = 0.85    # 85% from top (0.0-1.0)
```
- **Smaller box** = More strict, fewer false positives, requires precise hand placement
- **Larger box** = More lenient, easier to use, but may catch more false positives

### Stability Settings
```python
PREDICTION_BUFFER_SIZE = 3  # Number of consistent predictions needed (1-5 recommended)
```
- **Lower (1-2)** = Faster detection, but more false positives
- **Higher (4-5)** = Slower detection, but more accurate

### Confidence Threshold
```python
CONFIDENCE_THRESHOLD = 0.92  # Minimum confidence to accept (0.0-1.0)
```
- **Lower (0.85-0.90)** = More detections, but less accurate
- **Higher (0.93-0.98)** = Fewer false positives, but may miss some valid signs

### Cooldown Duration
```python
COOLDOWN_FRAMES = 15  # Frames to wait before re-detecting same letter
```
- **Lower (5-10)** = Can detect same letter faster, but may get duplicates
- **Higher (20-30)** = Prevents all duplicates, but slower for repeated letters

## Usage Tips

### For Best Results:
1. **Position your hand** inside the detection box (wait for green color)
2. **Hold the sign steady** for about 0.5 seconds
3. **Watch the confidence meter** - aim for values above 0.92
4. **Move deliberately** between signs to avoid transition errors
5. **Use the cooldown** - wait briefly between repeated letters

### Controls:
- **Spacebar**: Clear/reset the sentence and all buffers
- **Enter**: Perform grammar check (if enabled)
- **Q or close window**: Exit the application

## Troubleshooting

### Issue: Detection box too small/hard to keep hand inside
**Solution**: Increase box size by adjusting BOX_LEFT/RIGHT/TOP/BOTTOM values

### Issue: Signs detected too slowly
**Solution**: 
- Reduce `PREDICTION_BUFFER_SIZE` to 2
- Lower `CONFIDENCE_THRESHOLD` to 0.88
- Reduce `COOLDOWN_FRAMES` to 10

### Issue: Still getting false positives
**Solution**:
- Increase `PREDICTION_BUFFER_SIZE` to 4
- Increase `CONFIDENCE_THRESHOLD` to 0.95
- Make detection box smaller

### Issue: Not detecting signs even when hand is in box
**Solution**:
- Check if hand landmarks are being drawn (green lines)
- Lower `CONFIDENCE_THRESHOLD` to 0.85
- Ensure good lighting conditions
- Make sure 15+ hand landmarks are visible

## Technical Details

### Hand Detection Logic
- Requires at least 15 out of 21 hand landmarks to be inside the box
- Checks both left and right hands independently
- Clears prediction buffer when hand exits box

### Prediction Flow
1. Hand enters detection box → starts collecting keypoints
2. After 10 frames → makes prediction
3. Adds prediction to buffer (stores 3 most recent)
4. Checks if all 3 predictions are the same
5. Checks if average confidence ≥ threshold
6. If yes → adds to sentence and starts cooldown
7. Cooldown prevents re-detection for 15 frames

## Performance Impact

- **Minimal FPS drop**: ~1-2 frames per second due to additional checks
- **Memory usage**: Negligible (small prediction buffer)
- **CPU usage**: Slight increase for box boundary calculations

## Future Enhancements (Optional)

Consider adding:
- Multiple detection zones for different hand positions
- Adjustable box size via keyboard shortcuts
- Visual confidence bar instead of just text
- Sound feedback when sign is detected
- Recording mode to save detected sequences for debugging
