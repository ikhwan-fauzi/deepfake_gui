try:
    import mediapipe as mp
    print("Successfully imported mediapipe at the top of main.py")
except ImportError as e:
    print(f"Failed to import mediapipe at the top of main.py: {e}")
    # Optionally, re-raise or exit if this is critical for debugging
    # raise e 
    # import sys
    # sys.exit(1)
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from webcam_feed import WebcamFeed
from face_swap import FaceSwap

# --- Worker thread for processing frames ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self, webcam_feed, face_swapper):
        super().__init__()
        self._run_flag = True
        self.webcam_feed = webcam_feed
        self.face_swapper = face_swapper
        self.target_image_loaded = False
        # self.swapping_enabled_in_thread = False # Removed state

    def run(self):
        while self._run_flag:
            ret, frame = self.webcam_feed.get_frame()
            if not ret:
                self.error_signal.emit("Error: Could not read frame from webcam.")
                break
            
            # Attempt swap whenever target is loaded
            if self.target_image_loaded and self.face_swapper.target_image is not None:
                # swap_faces now contains the swapping logic
                processed_frame = self.face_swapper.swap_faces(frame)
            else:
                # If no target, just show webcam with landmarks if face is detected
                # Call swap_faces as it handles drawing landmarks as fallback
                processed_frame = self.face_swapper.swap_faces(frame)

            self.change_pixmap_signal.emit(processed_frame)
            
            # Add a small delay to control frame rate and reduce CPU usage
            # Aim for roughly 30 FPS (1000ms / 30fps = ~33ms per frame)
            # A small sleep helps prevent maxing out CPU even if processing is fast.
            QThread.msleep(10) # Sleep for 10 milliseconds

    def stop(self):
        self._run_flag = False
        self.wait()

    def set_target_loaded(self, loaded):
        self.target_image_loaded = loaded

    # def set_swapping_state(self, active): # Removed method
    #     self.swapping_enabled_in_thread = active

# --- Main Application Window ---
class DeepfakeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Deepfake GUI")
        self.setGeometry(100, 100, 1000, 560) # Adjusted size slightly

        # self.swapping_active = False # Removed state

        self._apply_stylesheet() # Apply custom styles

        # Initialize components
        try:
            self.webcam = WebcamFeed()
        except IOError as e:
            QMessageBox.critical(self, "Webcam Error", str(e))
            sys.exit(1) # Exit if webcam fails to initialize

        self.face_swapper = FaceSwap()

        # UI Elements
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main Horizontal Layout
        self.main_layout = QHBoxLayout(self.central_widget)

        # --- Left Panel (Target Image Upload and Preview) ---
        self.left_panel_widget = QWidget()
        self.left_panel_widget.setObjectName("left_panel_widget") # Set object name for styling
        self.left_layout = QVBoxLayout(self.left_panel_widget)
        self.left_layout.setContentsMargins(10, 10, 10, 10) # Add margins to left panel
        self.left_layout.setSpacing(10) # Add spacing between widgets in left panel
        
        self.upload_button = QPushButton("Upload Target Face Image", self)
        self.upload_button.clicked.connect(self.upload_target_image)
        self.left_layout.addWidget(self.upload_button)

        self.target_image_preview_label = QLabel("Target image preview") # Simpler text
        self.target_image_preview_label.setObjectName("target_image_preview_label") # Set object name
        self.target_image_preview_label.setAlignment(Qt.AlignCenter)
        self.target_image_preview_label.setFixedSize(320, 240) # Adjust size as needed
        # self.target_image_preview_label.setStyleSheet("border: 1px solid grey;") # Style handled by stylesheet
        self.left_layout.addWidget(self.target_image_preview_label)

        self.target_filename_label = QLabel("No target loaded") # Shorter text
        self.target_filename_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.target_filename_label)

        # Remove the swap button
        # self.swap_button = QPushButton("Start Face Swap", self)
        # self.swap_button.clicked.connect(self.toggle_face_swap)
        # self.left_layout.addWidget(self.swap_button)
        
        self.delete_button = QPushButton("Hapus Foto", self) # New Delete Button
        self.delete_button.clicked.connect(self.clear_target_photo)
        self.left_layout.addWidget(self.delete_button)

        self.left_layout.addStretch() # Pushes elements to the top

        # Adjust stretch factors: Give video panel more space (e.g., 3 vs 1)
        self.main_layout.addWidget(self.left_panel_widget, 1) # Proportion 1

        # --- Right Panel (Webcam/Output Feed) ---
        self.video_label = QLabel(self) # This is where the webcam/swapped feed goes
        self.video_label.setObjectName("video_label") # Set object name
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(640, 480) # Fixed size for the webcam feed
        # self.video_label.setStyleSheet("border: 1px solid black;") # Style handled by stylesheet
        self.main_layout.addWidget(self.video_label, 3) # Proportion 3 (larger)
        
        # Setup video thread
        self.thread = VideoThread(self.webcam, self.face_swapper)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.error_signal.connect(self.show_error_message)
        self.thread.start()

    def upload_target_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Target Image", "",
                                                   "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                print(f"Attempting to load target image: {file_path}")
                if self.face_swapper.load_target_image(file_path):
                    self.target_filename_label.setText(f"Target: {file_path.split('/')[-1]}")
                    self.thread.set_target_loaded(True)
                    
                    # Display target image preview
                    if self.face_swapper.target_image is not None:
                        preview_pixmap = self.convert_cv_qt(self.face_swapper.target_image, 
                                                            self.target_image_preview_label.width(), 
                                                            self.target_image_preview_label.height())
                        self.target_image_preview_label.setPixmap(preview_pixmap)
                    QMessageBox.information(self, "Success", "Target image loaded successfully and face detected.")
                else:
                    # load_target_image already prints detailed errors
                    self.target_filename_label.setText("Failed to load target. See console for details.")
                    self.target_image_preview_label.setText("Failed to load target preview.")
                    self.target_image_preview_label.setPixmap(QPixmap())
                    self.thread.set_target_loaded(False)
                    QMessageBox.warning(self, "Error", "Could not process target image. Face not detected or other error. Check console.")
            except Exception as e:
                print(f"An unexpected error occurred during target image loading: {e}")
                self.target_filename_label.setText("Error during image load. See console.")
                self.target_image_preview_label.setText("Error loading preview.")
                self.target_image_preview_label.setPixmap(QPixmap())
                self.thread.set_target_loaded(False)
                QMessageBox.critical(self, "Critical Error", f"An unexpected error occurred: {e}")

    def update_image(self, cv_img):
        """Updates the video_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img, self.video_label.width(), self.video_label.height())
        self.video_label.setPixmap(qt_img)

    def clear_target_photo(self):
        """Clears the loaded target image and stops swapping."""
        print("Clearing target photo.")
        self.face_swapper.target_image = None
        self.face_swapper.target_landmarks = None
        self.face_swapper.target_triangles_indices = None
        self.target_filename_label.setText("No target loaded")
        self.target_image_preview_label.setText("Target image preview")
        self.target_image_preview_label.setPixmap(QPixmap()) # Clear the pixmap
        self.thread.set_target_loaded(False)


    def convert_cv_qt(self, cv_img, width=None, height=None):
        """Convert from an opencv image to QPixmap"""
        if cv_img is None: # Add check for None image
             return QPixmap()
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale if width and height are provided
        # Keep aspect ratio if one dimension is None
        label_w = self.video_label.width()
        label_h = self.video_label.height()

        if width and height:
             p = convert_to_Qt_format.scaled(width, height, Qt.KeepAspectRatio)
        elif width:
             p = convert_to_Qt_format.scaledToWidth(width, Qt.SmoothTransformation)
        elif height:
             p = convert_to_Qt_format.scaledToHeight(height, Qt.SmoothTransformation)
        else:
             # Scale to fit the label while keeping aspect ratio
             p = convert_to_Qt_format.scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        return QPixmap.fromImage(p)

    def _apply_stylesheet(self):
        """Applies a basic stylesheet to the application."""
        style = """
            QMainWindow {
                background-color: #f0f0f0; /* Light grey background */
            }
            QWidget#left_panel_widget { /* Target specific widget by object name */
                 background-color: #e8e8e8;
                 border-radius: 5px;
            }
            QLabel {
                font-size: 11pt; /* Slightly larger default font */
                color: #333;
            }
            QLabel#video_label, QLabel#target_image_preview_label { /* Target specific labels */
                background-color: #ffffff; /* White background for image areas */
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #dcdcdc; /* Light grey button */
                border: 1px solid #b0b0b0;
                padding: 8px 15px; /* More padding */
                border-radius: 4px; /* Rounded corners */
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #c8c8c8; /* Darker grey on hover */
                border: 1px solid #909090;
            }
            QPushButton:pressed {
                background-color: #b0b0b0; /* Even darker when pressed */
            }
        """
        self.setStyleSheet(style)

    def show_error_message(self, message):
        QMessageBox.critical(self, "Application Error", message)
        # self.close_application() # Optionally close on critical errors

    # Remove toggle_face_swap method
    # def toggle_face_swap(self):
    #     if self.face_swapper.target_image is None: # Corrected check
    #         QMessageBox.warning(self, "No Target Image", "Please upload a target image before starting the swap.")
    #         return
    #
    #     self.swapping_active = not self.swapping_active
    #     if self.swapping_active:
    #         self.swap_button.setText("Stop Face Swap")
    #         print("Face swapping process enabled.")
    #     else:
    #         self.swap_button.setText("Start Face Swap")
    #         print("Face swapping process disabled.")
    #     self.thread.set_swapping_state(self.swapping_active)

    def closeEvent(self, event):
        """Handle window close event."""
        self.thread.stop()
        self.webcam.release()
        self.face_swapper.release()
        print("Application closing. Resources released.")
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Create assets directory if it doesn't exist for sample_target.jpg in face_swap.py
    import os
    if not os.path.exists('assets'):
        os.makedirs('assets')
        print("Created 'assets' directory.")
    
    # Create a dummy sample_target.jpg if it doesn't exist for face_swap.py example
    # This is more for the face_swap.py standalone test, but good to have for initial run
    if not os.path.exists('assets/sample_target.jpg'):
        dummy_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.putText(dummy_image, "TARGET", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite('assets/sample_target.jpg', dummy_image)
        print("Created dummy 'assets/sample_target.jpg'. Replace with a real face image for testing face_swap.py.")

    main_window = DeepfakeApp()
    main_window.show()
    sys.exit(app.exec_())
