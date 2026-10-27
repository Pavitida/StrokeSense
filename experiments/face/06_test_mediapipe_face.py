import mediapipe as mp

print("=" * 60)
print("MEDIAPIPE FACE CHECK")
print("=" * 60)

print("\nMediaPipe version:")
print(mp.__version__)

print("\nHas tasks:")
print(hasattr(mp, "tasks"))

print("\nHas solutions:")
print(hasattr(mp, "solutions"))

if hasattr(mp, "tasks"):
    print("\nMediaPipe Tasks API: AVAILABLE")

if hasattr(mp, "solutions"):
    print("MediaPipe Solutions API: AVAILABLE")

print("\nCHECK COMPLETE")