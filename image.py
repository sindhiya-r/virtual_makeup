import cv2
import numpy as np
import argparse
from utils import *

face_elements = [
    "LIP_LOWER",
    "LIP_UPPER",
    "EYEBROW_LEFT",
    "EYEBROW_RIGHT",
    "EYELINER_LEFT",
    "EYELINER_RIGHT",
    "EYESHADOW_LEFT",
    "EYESHADOW_RIGHT",
]

colors_map = {
    "LIP_UPPER": [0, 0, 255],      
    "LIP_LOWER": [0, 0, 255],
    "EYELINER_LEFT": [139, 0, 0], 
    "EYELINER_RIGHT": [139, 0, 0],
    "EYESHADOW_LEFT": [0, 100, 0], 
    "EYESHADOW_RIGHT": [0, 100, 0],
    "EYEBROW_LEFT": [19, 69, 139], 
    "EYEBROW_RIGHT": [19, 69, 139],
}

def get_skin_tone(image, landmarks):
    try:
        sample_point = landmarks.get(10, (image.shape[1]//2, image.shape[0]//2))
        b, g, r = image[sample_point[1], sample_point[0]]
        hsv_pixel = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv_pixel
        if v >= 200:
            return "fair"
        elif v >= 120:
            return "medium"
        else:
            return "deep"
    except:
        return "medium"

def adjust_makeup_colors(skin_tone):
    global colors_map
    if skin_tone == "fair":
        colors_map["LIP_LOWER"] = [139, 0, 0]  
        colors_map["LIP_UPPER"] = [139, 0, 0]
    elif skin_tone == "medium":
        colors_map["LIP_LOWER"] = [0, 0, 255]     
        colors_map["LIP_UPPER"] = [0, 0, 255]
    else:  
        colors_map["LIP_LOWER"] = [0, 0, 139]     
        colors_map["LIP_UPPER"] = [0, 0, 139]

def main(image_path):
    image = cv2.imread(image_path)
    face_landmarks = read_landmarks(image=image)
    if not face_landmarks:
        print("No face landmarks detected.")
        return
    skin_tone = get_skin_tone(image, face_landmarks)
    adjust_makeup_colors(skin_tone)
    face_connections = [face_points[idx] for idx in face_elements]
    colors = [colors_map[idx] for idx in face_elements]
    mask = np.zeros_like(image)
    mask = add_mask(
        mask,
        idx_to_coordinates=face_landmarks,
        face_connections=face_connections,
        colors=colors,
    )

    output = cv2.addWeighted(image, 1.0, mask, 0.2, 1.0)

    cv2.putText(output, f"Skin Tone: {skin_tone}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    show_image(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image to add Facial makeup")
    parser.add_argument("--img", type=str, help="Path to the image.")
    args = parser.parse_args()
    main(args.img)