import cv2
import numpy as np
from object_detection import ObjectDetection
import math
import os

# Input and output paths
video_path = "videos/video.mp4"
output_path = "output/tracking_obj_video.mp4"

# Ensure the output directory exists
os.makedirs("output", exist_ok=True)

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {os.path.abspath(video_path)}")
    exit()

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    print(f"Error: Could not open the video file at {video_path}. Please check the file format and codecs.")
    exit()

# Initialize Object Detection
od = ObjectDetection()

# Get video properties for saving
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter for MP4
output_video = cv2.VideoWriter(
    output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
)

if not output_video.isOpened():
    print(f"Error: Failed to initialize the video writer for output file {output_path}.")
    exit()

print(f"Processing video: {os.path.abspath(video_path)}")
print(f"Saving output to: {os.path.abspath(output_path)}")

# Initialize variables
count = 0
center_points_prev_frame = []
tracking_objects = {}
track_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    count += 1
    center_points_cur_frame = []  # Store center points of the current frame

    # Detect objects on the frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box

        # Calculate center points of the bounding box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))

        # Draw rectangle around detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Update tracking objects
    for pt in center_points_cur_frame:
        same_object_detected = False
        for object_id, prev_pt in tracking_objects.items():
            distance = math.hypot(prev_pt[0] - pt[0], prev_pt[1] - pt[1])

            if distance < 35:  # Threshold distance
                tracking_objects[object_id] = pt
                same_object_detected = True
                break

        # Assign new ID to new object
        if not same_object_detected:
            tracking_objects[track_id] = pt
            track_id += 1

    # Draw tracking points and IDs
    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            str(object_id),
            (pt[0] - 10, pt[1] - 10),  # Offset for better visibility
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    # Write the processed frame to the output video
    output_video.write(frame)

    # Prepare for next frame
    center_points_prev_frame = center_points_cur_frame.copy()

    # Uncomment to display the video while processing
    # cv2.imshow("Tracking", frame)
    # if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit early
    #     break

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()

print("Video processing complete.")
