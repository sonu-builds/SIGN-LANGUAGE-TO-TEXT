import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("Sign Language to Text")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create a class to process the video frames
class HandTrackingTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.tip_ids = [4, 8, 12, 16, 20]

    def transform(self, frame):
        # Convert the WebRTC frame to an OpenCV image
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(rgb)
        text = "?"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                lm_list = []
                h, w, c = img.shape
                for id, lm in enumerate(hand_landmarks.landmark):
                    lm_list.append((int(lm.x * w), int(lm.y * h)))

                if lm_list:
                    fingers = []
                    # Thumb
                    if lm_list[self.tip_ids[0]][0] > lm_list[self.tip_ids[0]-1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # Other fingers
                    for i in range(1, 5):
                        if lm_list[self.tip_ids[i]][1] < lm_list[self.tip_ids[i]-2][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    # Gesture → Text mapping
                    if fingers == [0,1,0,0,0]: text = "A"
                    elif fingers == [0,1,1,0,0]: text = "B"
                    elif fingers == [0,1,1,1,0]: text = "C"
                    elif fingers == [0,1,1,1,1]: text = "D"
                    elif fingers == [1,1,1,1,1]: text = "E"

        # Draw the text on the frame
        cv2.putText(img, f'Text: {text}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        return img

# Start the WebRTC streamer
webrtc_streamer(key="sign_language", video_transformer_factory=HandTrackingTransformer)
