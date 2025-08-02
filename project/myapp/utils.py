import cv2
import numpy as np

def extract_orb_features(image_path):
    img = cv2.imread(image_path, 0)  # grayscale
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    if descriptors is not None:
        return descriptors.tobytes()
    return None

def match_orb_features(des1_bytes, des2_bytes):
    try:
        des1 = np.frombuffer(des1_bytes, dtype=np.uint8).reshape(-1, 32)
        des2 = np.frombuffer(des2_bytes, dtype=np.uint8).reshape(-1, 32)
    except:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)
