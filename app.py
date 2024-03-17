from flask import Flask, Response
import cv2
import mediapipe as mp
import datetime
import socket

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection  # Import face detection module

# Initialize MediaPipe models
hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=2)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)  # Initialize face detection model

def track_hands(frame):
    results = hands.process(frame)
    detected_hands = 0
    frame_height, frame_width, _ = frame.shape
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            detected_hands += 1
            min_x = min(landmark.x * frame_width for landmark in hand_landmarks.landmark)
            max_x = max(landmark.x * frame_width for landmark in hand_landmarks.landmark)
            min_y = min(landmark.y * frame_height for landmark in hand_landmarks.landmark)
            max_y = max(landmark.y * frame_height for landmark in hand_landmarks.landmark)
            
            # Check if the hand is fully within the frame
            if min_x > 0 and max_x < frame_width and min_y > 0 and max_y < frame_height:
                for landmark in hand_landmarks.landmark:
                    cx, cy = int(landmark.x * frame_width), int(landmark.y * frame_height)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    
    # If no hands or only one hand is detected, display a message
    if detected_hands == 0:
        cv2.putText(frame, "no hands detected", (470, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    elif detected_hands == 1:
        cv2.putText(frame, "one hand detected", (470, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    elif detected_hands == 2:
        cv2.putText(frame, "2 hands detected", (470, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    elif detected_hands == 3:
        cv2.putText(frame, "3 hands??? what???", (470, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# Initialize variables for position smoothing
smoothed_x = {}
smoothed_y = {}
alpha_pos = 0.2  # Position smoothing factor (adjust as needed)

# Initialize variables for size smoothing
smoothed_w = {}
smoothed_h = {}
alpha_size = 0.2  # Size smoothing factor (adjust as needed)


def detect_faces(frame):
    global smoothed_x, smoothed_y, smoothed_w, smoothed_h
    
    results = face_detection.process(frame)
    if results.detections:
        for i, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Initialize smoothed coordinates if not already initialized
            if i not in smoothed_x:
                smoothed_x[i] = x
                smoothed_y[i] = y
                smoothed_w[i] = w
                smoothed_h[i] = h
            
            # Apply position smoothing
            smoothed_x[i] = alpha_pos * x + (1 - alpha_pos) * smoothed_x[i]
            smoothed_y[i] = alpha_pos * y + (1 - alpha_pos) * smoothed_y[i]
            
            # Apply size smoothing
            smoothed_w[i] = alpha_size * w + (1 - alpha_size) * smoothed_w[i]
            smoothed_h[i] = alpha_size * h + (1 - alpha_size) * smoothed_h[i]
            
            # Convert smoothed coordinates and size to integers
            smoothed_x_int = int(smoothed_x[i])
            smoothed_y_int = int(smoothed_y[i])
            smoothed_w_int = int(smoothed_w[i])
            smoothed_h_int = int(smoothed_h[i])
            
            cv2.rectangle(frame, (smoothed_x_int, smoothed_y_int), (smoothed_x_int + smoothed_w_int, smoothed_y_int + smoothed_h_int), (255, 0, 0), 2)
            cv2.putText(frame, "face", (smoothed_x_int, smoothed_y_int - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)





def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        else:
            frame = cv2.resize(frame, (640, 480))
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            ip_address = socket.gethostbyname(socket.gethostname())
            cv2.putText(frame, f'AR camera app', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f'time: {current_time}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f'ip: {ip_address}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            track_hands(frame)  # Call hand tracking function

            detect_faces(frame)  # Call face tracking function    
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AR Camera App</title>
    </head>
    <body>
        <h1>AR Camera App</h1>
        <img src="/video_feed" width="640" height="480">
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True, ssl_context=('cert.pem', 'key.pem'))
