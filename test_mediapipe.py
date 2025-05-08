# test_mediapipe.py
try:
    import mediapipe as mp
    print("MediaPipe imported successfully!")

    # Coba gunakan salah satu solusi, misalnya Face Detection
    mp_face_detection = mp.solutions.face_detection
    print("Face Detection module loaded.")

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        print("FaceDetection initialized successfully.")

except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")