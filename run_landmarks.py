"""Hand landmark detection on a single image using MediaPipe Tasks.

Based on:
https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python
"""

import mediapipe as mp
import cv2
import numpy as np

MODEL_PATH = "videoRetrieval/models/hand_landmarker.task"
IMAGE_PATH = "videoRetrieval/examples/image_1.png"
OUTPUT_PATH = "run_landmarks_output.png"


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in hand_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        height, width, _ = annotated_image.shape
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coords) * width)
        text_y = int(min(y_coords) * height) - 10

        cv2.putText(
            annotated_image,
            handedness[0].category_name,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (88, 205, 54),
            1,
            cv2.LINE_AA,
        )

    return annotated_image


def main():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE)

    with HandLandmarker.create_from_options(options) as landmarker:
        image = mp.Image.create_from_file(IMAGE_PATH)
        hand_landmarker_result = landmarker.detect(mp_image)

    
    detection_result = landmarker.detect(image)

    print(f"Detected {len(detection_result.hand_landmarks)} hand(s).")
    for i, handedness in enumerate(detection_result.handedness):
        print(f"  Hand {i}: {handedness[0].category_name} ({handedness[0].score:.2f})")

    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print(f"Saved annotated image to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
