from statistics import mode
import cv2
import mediapipe as mp
import time

WEBCAM_NUM = 0  # Windows number assigned to the webcam

mp_hands = mp.solutions.hands
mp_drawing_utils = mp.solutions.drawing_utils


class HandDetector():

    def __init__(self, max_hands=2, detection_con=0.5, track_con=0.5):
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        self.hand_detector = mp_hands.Hands(
            max_num_hands=self.max_hands,
            model_complexity=1,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con,
        )
        self.hand_results = None    # Store the processed results from Hands.process()

    def find_hands(self, img, draw_landmark=True):
        """Detect the hand in the bgr image `img` and draw the landmark on it by default"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hand_results = self.hand_detector.process(img_rgb)
        self.hand_results = hand_results.multi_hand_landmarks
        if self.hand_results:
            for hand in self.hand_results:
                for landmark_id, landmark_normalised_coord in enumerate(hand.landmark):
                    if draw_landmark:
                        mp_drawing_utils.draw_landmarks(
                            img, hand, mp_hands.HAND_CONNECTIONS)

    def retrieve_positions(self, img, hand_id=0):
        landmarks = []
        
        img_h, img_w, _ = img.shape
        if self.hand_results:
            for landmark_normalised_coord in self.hand_results[hand_id].landmark:
                landmarks.append(
                    (int(landmark_normalised_coord.x * img_w),
                     int(landmark_normalised_coord.y * img_h))
                )

        return landmarks


def main(): 
    
    # fps counter
    prev_time = 0

    cap = cv2.VideoCapture(WEBCAM_NUM)

    #
    hand_detector = HandDetector()

    while True:
        success, img = cap.read()
        hand_detector.find_hands(img, draw_landmark=True)
        hand_pos_pixel = hand_detector.retrieve_positions(img)
        if len(hand_pos_pixel) > 0:
            print(hand_pos_pixel[mp_hands.HandLandmark.THUMB_TIP])

        # print fps
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, str(int(fps)), (10, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255))


        # display image and overlay info
        cv2.imshow("Hand Track Webcam", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
