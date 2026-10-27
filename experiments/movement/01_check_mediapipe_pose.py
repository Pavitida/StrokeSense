import mediapipe as mp

print("=" * 70)
print("MOVEMENT STEP 1 - MEDIAPIPE POSE CHECK")
print("=" * 70)

print("\nMediaPipe version:")
print(mp.__version__)

print("\nHas tasks:")
print(hasattr(mp, "tasks"))

if not hasattr(mp, "tasks"):
    raise RuntimeError(
        "MediaPipe Tasks API is not available."
    )

print("\nMediaPipe Tasks API: AVAILABLE")

# Check vision module
try:
    from mediapipe.tasks.python import vision

    print("MediaPipe Vision API: AVAILABLE")

    print("\nPoseLandmarker:")
    print(hasattr(vision, "PoseLandmarker"))

    if not hasattr(vision, "PoseLandmarker"):
        raise RuntimeError(
            "PoseLandmarker is not available."
        )

    print("\nPose Landmarker API: AVAILABLE")

except Exception as e:
    print("\nERROR:")
    print(e)
    raise

print("\n" + "=" * 70)
print("MOVEMENT STEP 1 COMPLETE")
print("=" * 70)