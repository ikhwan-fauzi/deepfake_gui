import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# --- Constants for Eye Aspect Ratio (EAR) ---
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR_INDICES = [33, 160, 158, 133, 144, 153]
EYE_AR_THRESH = 0.20

def calculate_ear(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C + 1e-6)
    return ear

# Helper function to calculate triangle area
def triangle_area(pt1, pt2, pt3):
    return 0.5 * abs(pt1[0]*(pt2[1]-pt3[1]) + pt2[0]*(pt3[1]-pt1[1]) + pt3[0]*(pt1[1]-pt2[1]))

class FaceSwap:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.stream_face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.static_face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.3, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.target_image = None
        self.target_landmarks = None
        self.target_triangles_indices = None

    def load_target_image(self, image_path):
        self.target_image = cv2.imread(image_path)
        if self.target_image is None:
            print(f"Error: Could not load target image from {image_path}")
            return False
        results = self._get_landmarks(self.target_image, is_target_image=True)
        if results and results.multi_face_landmarks:
            self.target_landmarks = results.multi_face_landmarks[0]
            print(f"Successfully detected {len(results.multi_face_landmarks)} face(s) in target image {image_path}.")
            # --- Precompute Triangulation ---
            target_landmarks_list = []
            th, tw, _ = self.target_image.shape
            for lm in self.target_landmarks.landmark:
                x, y = int(lm.x * tw), int(lm.y * th)
                target_landmarks_list.append((x, y))
            target_points = np.array(target_landmarks_list, dtype=np.float32)
            if len(target_points) < 4:
                 print("Error: Not enough landmarks detected in target image for triangulation.")
                 self.target_image = None; self.target_landmarks = None; self.target_triangles_indices = None
                 return False
            rect_target = (0, 0, tw, th)
            subdiv_target = cv2.Subdiv2D(rect_target)
            try:
                for p in target_points:
                    px, py = int(p[0]), int(p[1])
                    if 0 <= px < tw and 0 <= py < th: subdiv_target.insert((px, py))
                    else: print(f"Warning: Skipping landmark point outside image bounds during triangulation: ({px}, {py})")
                triangles_coords = subdiv_target.getTriangleList()
            except cv2.error as e_tri:
                 print(f"Error during target triangulation: {e_tri}")
                 self.target_image = None; self.target_landmarks = None; self.target_triangles_indices = None
                 return False
            self.target_triangles_indices = []
            for tri_coords in triangles_coords:
                pt1, pt2, pt3 = (tri_coords[0], tri_coords[1]), (tri_coords[2], tri_coords[3]), (tri_coords[4], tri_coords[5])
                indices = []
                for pt_tri in [pt1, pt2, pt3]:
                    found_idx = -1; min_dist_sq = float('inf')
                    for i, pt_orig in enumerate(target_points):
                        dist_sq = (pt_tri[0] - pt_orig[0])**2 + (pt_tri[1] - pt_orig[1])**2
                        if dist_sq < min_dist_sq and dist_sq < 1.0: min_dist_sq = dist_sq; found_idx = i
                    if found_idx != -1: indices.append(found_idx)
                    else: indices = []; break
                if len(indices) == 3:
                    pts_for_area = target_points[list(indices)]
                    if triangle_area(pts_for_area[0], pts_for_area[1], pts_for_area[2]) > 0.1:
                        if 0 <= pt1[0] < tw and 0 <= pt1[1] < th and 0 <= pt2[0] < tw and 0 <= pt2[1] < th and 0 <= pt3[0] < tw and 0 <= pt3[1] < th:
                            self.target_triangles_indices.append(tuple(indices))
            if not self.target_triangles_indices:
                print("Error: Failed to map any triangles to landmark indices.")
                self.target_image = None; self.target_landmarks = None
                return False
            # --- End Precompute Triangulation ---

        else:
            print(f"Error: Could not detect face in target image {image_path}.")
            if results:
                print(f"  MediaPipe results object for target image: {type(results)}")
            else:
                print(f"  MediaPipe results object for target image was None.")
            self.target_image = None # Invalidate if no face found
            self.target_landmarks = None
            self.target_triangles_indices = None
            return False
        return True

    def _get_landmarks(self, image, is_target_image=False):
        """
        Detects face landmarks in a given image.
        :param image: The input image (NumPy array).
        :param is_target_image: Boolean, True if processing the target image, False for stream.
        :return: MediaPipe's face mesh results.
        """
        # Convert the BGR image to RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image_rgb.flags.writeable = False

        if is_target_image:
            results = self.static_face_mesh.process(image_rgb)
        else:
            results = self.stream_face_mesh.process(image_rgb)

        image_rgb.flags.writeable = True
        return results

    def swap_faces(self, source_frame):
        """
        Performs the face swap operation on the source_frame using the loaded target_image.
        :param source_frame: The webcam frame (NumPy array).
        :return: The frame with the swapped face or original frame if swap fails.
        """
        frame_processed = source_frame.copy()
        source_results = self._get_landmarks(frame_processed, is_target_image=False) # For webcam stream
        output_frame = frame_processed.copy() # Work on a copy

        source_points = None # Initialize source_points to None
        clone_successful = False # Flag to track if seamlessClone succeeded

        if source_results and source_results.multi_face_landmarks and \
           self.target_image is not None and self.target_landmarks is not None and \
           self.target_triangles_indices is not None: # Check for precomputed indices

            source_face_landmarks_mp = source_results.multi_face_landmarks[0] # Assuming one face

            # Convert MediaPipe landmarks to a list of (x, y) tuples
            source_landmarks_list = []
            for lm in source_face_landmarks_mp.landmark:
                ih, iw, _ = output_frame.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                source_landmarks_list.append((x, y))

            target_landmarks_list = []
            for lm in self.target_landmarks.landmark: # self.target_landmarks is already loaded
                th, tw, _ = self.target_image.shape
                x, y = int(lm.x * tw), int(lm.y * th)
                target_landmarks_list.append((x, y))

            source_points = np.array(source_landmarks_list, dtype=np.float32) # Assign source_points here
            target_points = np.array(target_landmarks_list, dtype=np.float32)

            if len(source_points) != len(target_points) or len(source_points) < 68: # Check for sufficient landmarks
                print("[SWAP_LOGIC] Landmark point mismatch or insufficient points. Swap skipped.")
                # No fallback drawing here
            else:
                # Create a canvas for the warped target face
                warped_target_face_canvas = np.zeros_like(output_frame, dtype=output_frame.dtype)

                # --- Use precomputed triangle indices ---
                if not self.target_triangles_indices:
                     print("[SWAP_LOGIC] Precomputed target_triangles_indices is empty. Swap skipped.")
                     # No fallback drawing here
                else:
                    triangles_processed_count = 0
                    for indices in self.target_triangles_indices: # Iterate through triplets of indices
                        try:
                            # Get points for the current triangle from target and source using indices
                            triangle_target_pts_np = target_points[list(indices)]
                            triangle_source_pts_np = source_points[list(indices)]
                        except IndexError:
                            # This might happen if landmark detection fluctuates frame-to-frame
                            print(f"[SWAP_LOGIC] IndexError accessing source_points for indices {indices}. Skipping triangle.") # Ensure this is uncommented
                            continue # Skip this triangle

                        # Crop the triangle from the target image
                        rect_target_crop = cv2.boundingRect(triangle_target_pts_np)
                        x_t, y_t, w_t, h_t = rect_target_crop
                        if w_t <= 0 or h_t <= 0:
                            # print(f"  Skipping target triangle {indices} due to zero area bbox ({w_t}x{h_t}).") # Optional: very verbose
                            continue

                        cropped_triangle_target_img = self.target_image[y_t : y_t + h_t, x_t : x_t + w_t]
                        # --- Add check for valid crop ---
                        if cropped_triangle_target_img.size == 0:
                            # print(f"  Skipping target triangle {indices} due to empty crop.") # Optional verbose log
                            continue
                        # --- End check ---
                        triangle_target_pts_relative = triangle_target_pts_np - np.array([x_t, y_t], dtype=np.float32)

                        # Define the bounding box for the source triangle
                        rect_source_crop = cv2.boundingRect(triangle_source_pts_np)
                        x_s, y_s, w_s, h_s = rect_source_crop
                        if w_s <= 0 or h_s <= 0:
                            # print(f"  Skipping source triangle {indices} due to zero area bbox ({w_s}x{h_s}).") # Optional: very verbose
                            continue

                        triangle_source_pts_relative = triangle_source_pts_np - np.array([x_s, y_s], dtype=np.float32)

                        # Affine transform
                        warp_matrix = cv2.getAffineTransform(triangle_target_pts_relative, triangle_source_pts_relative)
                        warped_cropped_triangle_source = cv2.warpAffine(
                            cropped_triangle_target_img, warp_matrix, (w_s, h_s),
                            None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                        )

                        # Create a mask for this triangle on the source canvas
                        mask_triangle_source = np.zeros((h_s, w_s), dtype=np.uint8)
                        cv2.fillConvexPoly(mask_triangle_source, np.int32(triangle_source_pts_relative), 255)

                        # Place the warped triangle onto the warped_target_face_canvas
                        # Ensure ROI is within bounds of warped_target_face_canvas
                        if y_s + h_s <= warped_target_face_canvas.shape[0] and \
                           x_s + w_s <= warped_target_face_canvas.shape[1] and \
                           y_s >= 0 and x_s >= 0:
                            roi_on_canvas = warped_target_face_canvas[y_s : y_s + h_s, x_s : x_s + w_s]
                            # Apply mask
                            masked_warped_triangle = cv2.bitwise_and(warped_cropped_triangle_source, warped_cropped_triangle_source, mask=mask_triangle_source)

                            # Create an inverse mask for the ROI on canvas
                            roi_mask_inv = cv2.bitwise_not(mask_triangle_source)
                            roi_bg = cv2.bitwise_and(roi_on_canvas, roi_on_canvas, mask=roi_mask_inv)

                            # Add the masked warped triangle to the ROI background
                            dst_triangle_area = cv2.add(roi_bg, masked_warped_triangle)
                            warped_target_face_canvas[y_s : y_s + h_s, x_s : x_s + w_s] = dst_triangle_area
                            triangles_processed_count +=1

                    if triangles_processed_count == 0:
                         print("[SWAP_LOGIC] WARNING: Processed 0 triangles successfully in the loop. Check for degenerate triangles or bounding box issues.")

                    # Seamless blending
                    # Create a mask cropped to the face hull's bounding box
                    source_face_hull = cv2.convexHull(source_points.astype(int))
                    r = cv2.boundingRect(source_face_hull) # (x, y, w, h) of the hull
                    center = (r[0] + r[2] // 2, r[1] + r[3] // 2)
                    source_face_mask_cropped = np.zeros((r[3], r[2]), dtype=np.uint8) # h, w

                    # Adjust hull points to be relative to the cropped mask's origin (r[0], r[1])
                    hull_relative = source_face_hull - np.array([r[0], r[1]])
                    cv2.fillConvexPoly(source_face_mask_cropped, hull_relative, 255)

                    # --- Mouth hole removed as per request ---

                    # --- Apply Gaussian Blur to Mask ---
                    # Kernel size should be odd. Increase size for more blur.
                    source_face_mask_blurred = cv2.GaussianBlur(source_face_mask_cropped, (11, 11), 0) # Increased blur

                    # Center remains the same (center of the hull in the original frame)

                    # Get dimensions for safety checks (still use full frame dimensions)
                    frame_h, frame_w = output_frame.shape[:2]
                    mask_h, mask_w = source_face_mask_blurred.shape[:2] # Use cropped mask dimensions

                    # Check if the center point allows the mask to be placed without going out of bounds
                    clone_ready = True
                    if not (0 <= center[0] < frame_w and 0 <= center[1] < frame_h):
                        print(f"[SWAP_LOGIC] Center point {center} is out of destination frame bounds ({frame_w}x{frame_h}). Skipping clone.")
                        clone_ready = False

                    # Further check: The mask defines the region to be cloned.
                    if clone_ready: # Only check further if center is okay
                        paste_tl_x = center[0] - mask_w // 2 # Use mask width
                        paste_tl_y = center[1] - mask_h // 2 # Use mask height
                        paste_br_x = paste_tl_x + mask_w
                        paste_br_y = paste_tl_y + mask_h
                        # Add a small safety margin (e.g., 2 pixels)
                        margin = 2
                        if not (paste_tl_x >= margin and paste_tl_y >= margin and \
                                paste_br_x <= frame_w - margin and paste_br_y <= frame_h - margin):
                            print(f"[SWAP_LOGIC] Effective paste area for hull (with margin {margin}px) is out of destination frame bounds. Skipping clone.")
                            clone_ready = False

                    # Also ensure the source canvas (warped_target_face_canvas) has content
                    if clone_ready and not np.any(warped_target_face_canvas):
                        print("[SWAP_LOGIC] warped_target_face_canvas is empty. Skipping seamlessClone.")
                        clone_ready = False

                    if clone_ready:
                        try:
                            # Crop the warped source canvas to the bounding box 'r'
                            # Ensure r is within the bounds of warped_target_face_canvas first
                            if r[1] >= 0 and r[1]+r[3] <= warped_target_face_canvas.shape[0] and \
                               r[0] >= 0 and r[0]+r[2] <= warped_target_face_canvas.shape[1]:

                                cropped_warped_canvas = warped_target_face_canvas[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]

                                # Ensure the cropped canvas and the mask have the same dimensions
                                if cropped_warped_canvas.shape[:2] == source_face_mask_blurred.shape:
                                    # print(f"[SWAP_LOGIC] Attempting seamlessClone with center: {center}") # Less verbose

                                    # --- Optional Placeholder: Add Color Correction Here ---
                                    # Before cloning, one could adjust colors of cropped_warped_canvas
                                    # to match the lighting/color of the output_frame region around 'center'.
                                    # This is complex (e.g., histogram matching on face ROIs).
                                    # corrected_canvas = color_correct(cropped_warped_canvas, output_frame, source_face_mask_blurred, center)
                                    # output_frame = cv2.seamlessClone(corrected_canvas, ...) # Use corrected canvas
                                    # --- End Optional Color Correction ---

                                    output_frame = cv2.seamlessClone(
                                        cropped_warped_canvas,     # Cropped Source
                                        output_frame,              # Destination
                                        source_face_mask_blurred,  # Cropped Mask (blurred)
                                        center,                    # Center (still in dest coordinates)
                                        cv2.MIXED_CLONE
                                    )
                                    clone_successful = True # Mark clone as successful
                                    # print("[SWAP_LOGIC] seamlessClone completed.") # Less verbose
                                else:
                                    print(f"[SWAP_LOGIC] Mismatch between cropped canvas ({cropped_warped_canvas.shape[:2]}) and mask ({source_face_mask_blurred.shape}). Skipping clone.")
                                    # No fallback drawing here
                            else:
                                 print(f"[SWAP_LOGIC] Bounding box 'r' {r} is out of bounds for warped_target_face_canvas. Skipping clone.")
                                 # No fallback drawing here

                        except cv2.error as e:
                            print(f"[SWAP_LOGIC] Error during seamlessClone: {e}. Swap skipped.")
                            # No fallback drawing here
                    else: # If not clone_ready initially
                        print("[SWAP_LOGIC] Conditions for seamlessClone not met. Swap skipped.")
                        # No fallback drawing here

            # --- Eye Blinking Logic --- Moved inside the main 'if' block ---
            # Run this only if source_points were successfully calculated
            if source_points is not None:
                try:
                    left_eye_pts = source_points[LEFT_EYE_EAR_INDICES]
                    right_eye_pts = source_points[RIGHT_EYE_EAR_INDICES]

                    left_ear = calculate_ear(left_eye_pts)
                    right_ear = calculate_ear(right_eye_pts)

                    # Average EAR or check both eyes
                    avg_ear = (left_ear + right_ear) / 2.0

                    # If eyes are closed in source frame (blink detected)
                    if avg_ear < EYE_AR_THRESH:
                        # print("[BLINK_LOGIC] Blink detected, pasting original eyes.")
                        # Paste original eyes back onto the potentially swapped frame

                        # Left Eye
                        left_eye_all_pts = source_points[LEFT_EYE_INDICES].astype(int)
                        left_eye_hull = cv2.convexHull(left_eye_all_pts)
                        left_eye_mask = np.zeros_like(output_frame[:,:,0])
                        cv2.fillConvexPoly(left_eye_mask, left_eye_hull, 255)
                        # Apply slight blur to mask edge
                        left_eye_mask = cv2.GaussianBlur(left_eye_mask, (5,5), 0)

                        # Right Eye
                        right_eye_all_pts = source_points[RIGHT_EYE_INDICES].astype(int)
                        right_eye_hull = cv2.convexHull(right_eye_all_pts)
                        right_eye_mask = np.zeros_like(output_frame[:,:,0])
                        cv2.fillConvexPoly(right_eye_mask, right_eye_hull, 255)
                        # Apply slight blur to mask edge
                        right_eye_mask = cv2.GaussianBlur(right_eye_mask, (5,5), 0)

                        # Combine masks
                        eye_mask = cv2.bitwise_or(left_eye_mask, right_eye_mask)

                        # Get original eye regions from source_frame (the input webcam frame)
                        original_eyes = cv2.bitwise_and(source_frame, source_frame, mask=eye_mask)

                        # Create inverse mask for background on output_frame (which might be swapped)
                        eye_mask_inv = cv2.bitwise_not(eye_mask)
                        output_bg = cv2.bitwise_and(output_frame, output_frame, mask=eye_mask_inv)

                        # Combine background with original eyes
                        output_frame = cv2.add(output_bg, original_eyes)

                except IndexError:
                    print("[BLINK_LOGIC] Warning: Could not get eye landmarks for EAR calculation.")
                except Exception as e_blink:
                    print(f"[BLINK_LOGIC] Warning: Error during blink processing: {e_blink}")
            # --- End Eye Blinking Logic ---

        elif source_results and source_results.multi_face_landmarks: # Only source face detected (no target or indices)
             # No fallback drawing here - show original frame
             pass
        else: # No source face detected at all
             pass # No face detected, just return the original output_frame

        return output_frame

    def release(self):
        """
        Releases resources used by MediaPipe.
        """
        self.stream_face_mesh.close()
        self.static_face_mesh.close()

if __name__ == '__main__':
    # Example Usage (requires a webcam and a target image)
    # Create an 'assets' folder and put 'sample_target.jpg' in it for this to run.

    # Create a dummy assets folder and sample image for testing if they don't exist
    import os
    if not os.path.exists('assets'):
        os.makedirs('assets')
    if not os.path.exists('assets/sample_target.jpg'):
        # Create a simple dummy image if none exists
        dummy_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.putText(dummy_image, "TARGET", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite('assets/sample_target.jpg', dummy_image)
        print("Created dummy 'assets/sample_target.jpg'. Replace with a real face image.")

    face_swapper = FaceSwap()
    if not face_swapper.load_target_image('assets/sample_target.jpg'):
        print("Failed to load target image. Exiting.")
        exit()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        exit()

    print("FaceSwap initialized. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        processed_frame = face_swapper.swap_faces(frame)
        cv2.imshow('Face Swap Test', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    face_swapper.release()
    cap.release()
    cv2.destroyAllWindows()
    print("FaceSwap released, webcam released, and windows closed.")