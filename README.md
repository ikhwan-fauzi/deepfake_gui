# Real-time Deepfake GUI (Python + PyQt5 + MediaPipe)

This project implements a real-time face swapping application using Python, PyQt5 for the GUI, OpenCV for image processing, and MediaPipe for face landmark detection. It allows users to capture video from their webcam and attempts to swap their face with a face from an uploaded target image in real-time.

## Project Structure

```
deepfake_gui/
├── main.py                  # Entry point + PyQt5 GUI
├── face_swap.py            # Core face swapping logic (MediaPipe landmarks, triangulation, blending)
├── webcam_feed.py          # Webcam handling class
├── utils.py                # Helper utilities (currently empty)
├── assets/
│   └── sample_target.jpg   # Example target face image (auto-generated if missing)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Features

- **Real-time Webcam Feed:** Displays the video stream from the default webcam.
- **Target Image Upload:** Allows users to upload a target image (JPG, PNG, etc.) containing the face they want to swap onto the webcam feed.
- **Automatic Face Swapping:** Once a target image with a detectable face is loaded, the application automatically attempts to perform the face swap on the webcam feed.
- **Face Detection & Landmarking:** Uses MediaPipe Face Mesh to detect faces and extract 478 landmarks for both the source (webcam) and target (uploaded image) faces.
- **Triangle-based Warping:** Precomputes Delaunay triangulation on the target face landmarks. In real-time, it maps corresponding triangles from the target face onto the source face using affine transformations.
- **Seamless Blending:** Uses OpenCV's `seamlessClone` (specifically `MIXED_CLONE`) to blend the warped target face onto the source frame, attempting to match lighting conditions.
- **Mouth Realism Attempt:** The blending mask includes the mouth area, transferring the target's mouth texture.
- **Blinking Realism Attempt:** Calculates Eye Aspect Ratio (EAR) for the source face. If a blink is detected (EAR below threshold), it attempts to paste the original closed eyes from the webcam feed back onto the swapped face.
- **Performance Optimizations:**
  - Target face triangulation is precomputed only once when the image is loaded.
  - Frame processing loop includes a small delay (`QThread.msleep`) to prevent excessive CPU usage.
- **GUI:** Built with PyQt5, featuring separate panels for target image preview/controls and the main video output. Includes basic styling.

## Setup

1.  **Clone the repository (or create the files as listed above).**

2.  **Create and activate a Python virtual environment:**
    _(Recommended to avoid dependency conflicts)_

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    - Ensure you have the necessary **Microsoft Visual C++ Redistributables** installed (both x86 and x64 versions are recommended on Windows). This is often required for `mediapipe`.
    - Install the Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare a target image (Optional but Recommended):**
    - Create an `assets` folder in the project root if it doesn't exist.
    - Place an image file (e.g., `my_target.jpg`) containing a clear, reasonably frontal face into the `assets` folder. Using a good target image significantly impacts the swap quality.
    - If `assets/sample_target.jpg` is not found when running for the first time, a dummy image will be created.

## How to Run

1.  Ensure your virtual environment is activated and dependencies are installed.
2.  Run the main application from the terminal:
    ```bash
    python main.py
    ```
3.  The GUI window will appear.
4.  Click "Upload Target Face Image" and select your desired target image file.
5.  If a face is detected in the target image, the application will immediately start attempting the real-time face swap on the video feed displayed in the right panel.
6.  Click "Hapus Foto" to clear the current target image and stop the swap attempt.
7.  Close the window or press Ctrl+C in the terminal to quit.

## Key Libraries Used

- **PyQt5:** For the graphical user interface.
- **OpenCV (cv2):** For image/video reading, writing, manipulations, warping, and blending (`seamlessClone`).
- **MediaPipe:** For fast and robust face detection and landmark extraction (Face Mesh).
- **NumPy:** For numerical operations, especially array manipulations.
- **SciPy:** Used for distance calculations (`scipy.spatial.distance`) in the Eye Aspect Ratio calculation.
