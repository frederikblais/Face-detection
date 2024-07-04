import cv2
import numpy as np
import time
from queue import Queue
from threading import Thread

# Load the face and eye detection classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_faces_and_eyes(frame, scale_factor=1.1, min_neighbors=3, min_size=(30, 30)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)
    
    confirmed_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) > 0:
            confirmed_faces.append((x, y, w, h))
    
    return confirmed_faces

def draw_boxes(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

def capture_frames(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)

class SimpleTracker:
    def __init__(self, max_frames=5):
        self.max_frames = max_frames
        self.faces = []
    
    def update(self, new_faces):
        if not self.faces:
            self.faces = [new_faces]
        else:
            self.faces.append(new_faces)
            if len(self.faces) > self.max_frames:
                self.faces.pop(0)
    
    def get_stable_faces(self):
        if not self.faces:
            return []
        all_faces = [face for frame in self.faces for face in frame]
        if not all_faces:
            return []
        mean_face = np.mean(all_faces, axis=0)
        return [tuple(map(int, mean_face))]

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_queue = Queue(maxsize=1)
    capture_thread = Thread(target=capture_frames, args=(cap, frame_queue))
    capture_thread.daemon = True
    capture_thread.start()
    
    prev_frame_time = 0
    new_frame_time = 0
    
    tracker = SimpleTracker()
    
    while True:
        if frame_queue.empty():
            continue
        
        frame = frame_queue.get()
        
        new_frame_time = time.time()
        
        faces = detect_faces_and_eyes(frame, scale_factor=1.05, min_neighbors=3, min_size=(20, 20))
        tracker.update(faces)
        stable_faces = tracker.get_stable_faces()
        
        result = draw_boxes(frame, stable_faces)
        
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(result, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Detection', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()