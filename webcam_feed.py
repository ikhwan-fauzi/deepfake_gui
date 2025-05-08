import cv2

class WebcamFeed:
    def __init__(self, camera_index=0):
        """
        Initializes the webcam feed.
        :param camera_index: The index of the camera to use (default is 0).
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

    def get_frame(self):
        """
        Captures a single frame from the webcam.
        :return: A tuple (ret, frame), where ret is True if the frame was captured successfully,
                 and frame is the captured image. Returns (False, None) if an error occurs.
        """
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """
        Releases the webcam.
        """
        self.cap.release()

if __name__ == '__main__':
    # Example usage:
    webcam = None
    try:
        webcam = WebcamFeed()
        print("Webcam initialized. Press 'q' to quit.")
        while True:
            ret, frame = webcam.get_frame()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            cv2.imshow('Webcam Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except IOError as e:
        print(e)
    finally:
        if webcam:
            webcam.release()
        cv2.destroyAllWindows()
        print("Webcam released and windows closed.")