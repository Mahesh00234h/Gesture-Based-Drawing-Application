import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize drawing canvas and variables
canvas = None
drawing = False  # Default to drawing disabled
eraser_mode = False  # Default to drawing mode
prev_x, prev_y = None, None  # Previous finger position
brush_size = 5  # Default brush size
color_index = 0  # Index for color selection
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # Red, Green, Blue, Yellow
history = []  # Stack to store previous canvas states

# Define utility functions
def detect_gesture(landmarks):
    """Detect gestures based on hand landmarks."""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Calculate distances
    thumb_index_dist = np.linalg.norm(np.array(thumb_tip[:2]) - np.array(index_tip[:2]))
    thumb_middle_dist = np.linalg.norm(np.array(thumb_tip[:2]) - np.array(middle_tip[:2]))
    index_middle_dist = np.linalg.norm(np.array(index_tip[:2]) - np.array(middle_tip[:2]))

    if thumb_index_dist < 0.05 and thumb_middle_dist < 0.05:
        return "fist"  # Eraser mode
    elif index_middle_dist > 0.1 and pinky_tip[1] < middle_tip[1] and pinky_tip[1] < ring_tip[1]:
        return "peace"  # Change color
    elif thumb_index_dist < 0.05:
        return "pinch"  # Adjust brush size
    elif landmarks[8][1] > landmarks[6][1] and landmarks[12][1] > landmarks[10][1]:
        return "open_palm"  # Clear canvas
    return "draw"

# Open the camera feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    h, w, c = frame.shape

    # Create a blank canvas if not initialized
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            gesture = detect_gesture(landmarks)

            if gesture == "fist":
                eraser_mode = True
                prev_x, prev_y = None, None
            elif gesture == "open_palm":
                history.append(canvas.copy())  # Save state before clearing
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                prev_x, prev_y = None, None
            elif gesture == "peace":
                color_index = (color_index + 1) % len(colors)
            elif gesture == "pinch":
                brush_size = max(1, brush_size + 1) if brush_size < 20 else 1
            elif gesture == "draw":
                eraser_mode = False
                index_x, index_y = int(landmarks[8][0] * w), int(landmarks[8][1] * h)
                if prev_x is not None and prev_y is not None:
                    if eraser_mode:
                        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (0, 0, 0), brush_size * 2)
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), colors[color_index], brush_size)
                prev_x, prev_y = index_x, index_y
            else:
                prev_x, prev_y = None, None
    else:
        prev_x, prev_y = None, None

    # Merge canvas and frame
    combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.putText(combined_frame, f"Color: {colors[color_index]} Brush: {brush_size}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Undo feature
    if cv2.waitKey(1) & 0xFF == ord('u') and history:
        canvas = history.pop()

    # Display the frames
    cv2.imshow("Gesture-Based Drawing", combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("drawing.png", canvas)
        print("Canvas saved as drawing.png")

cap.release()
cv2.destroyAllWindows()
