import cv2
import mediapipe as md

# MediaPipe modules for drawing
md_drawing = md.solutions.drawing_utils
md_drawing_styles = md.solutions.drawing_styles
md_pose = md.solutions.pose

count = 0  # Counter for push-ups
position = None  # Track the current position (None, "down", or "up")

cap = cv2.VideoCapture(0)  # Open the default camera

with md_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Empty camera frame")
            break  # Exit if no frame is captured

        # Flip the image horizontally and convert to RGB
        image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Pose
        result = pose.process(image_rgb)

        imlist = []  # List to store landmark IDs and coordinates

        if result.pose_landmarks and len(result.pose_landmarks.landmark) >= 15:  # Ensure enough landmarks are detected
            md_drawing.draw_landmarks(  # Draw the pose landmarks on the original image
                image, result.pose_landmarks, md_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=md_drawing_styles.get_default_pose_landmarks_style()
            )

            for id, lm in enumerate(result.pose_landmarks.landmark):
                h, w, _ = image.shape  # Image height and width
                x, y = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel values
                imlist.append([id, x, y])  # Append [ID, X, Y]

            # Check for "down" position: Elbows lower than shoulders
            if imlist[13][2] > imlist[11][2] and imlist[14][2] > imlist[12][
                2]:  # Left Elbow Y > Left Shoulder Y and Right Elbow Y > Right Shoulder Y
                position = "down"  # Set position to down

            # Check for "up" position: Elbows at or above shoulders, and previous position was down
            elif imlist[13][2] <= imlist[11][2] and imlist[14][2] <= imlist[12][2] and position == "down":
                position = "up"  # Set position to up
                count += 1  # Increment the count
                print(f"Push-up count: {count}")  # Print the current count

        # Display the image with the count overlaid
        cv2.putText(image, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Push-Up Counter", cv2.flip(image, 1))  # Show the flipped image

        key = cv2.waitKey(1)  # Wait for a key press
        if key == ord('q'):  # Quit if 'q' is pressed
            break

cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
