from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Fingerprint
import cv2
import numpy as np
import os


def extract_orb_features(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        return None
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return descriptors


class AddFingerprintView(APIView):
    def post(self, request):
        name = request.data.get('name')
        image_path = request.data.get('image_path')

        if not name or not image_path:
            return Response({'error': 'Name and image_path required'}, status=400)

        if not os.path.exists(image_path):
            return Response({'error': 'Image path does not exist'}, status=400)

        descriptors = extract_orb_features(image_path)

        if descriptors is None:
            return Response({'error': 'No features found in image'}, status=400)

        fingerprint = Fingerprint(name=name)
        fingerprint.image.save(os.path.basename(image_path), open(image_path, 'rb'))
        fingerprint.descriptors = descriptors.tobytes()
        fingerprint.save()

        return Response({'message': 'Fingerprint added successfully'}, status=201)


class MatchFingerprintView(APIView):
    def post(self, request, *args, **kwargs):
        image_path = request.data.get('image_path')
        if not image_path or not os.path.exists(image_path):
            return Response({"error": "Image not found in request or path invalid"}, status=400)

        input_des = extract_orb_features(image_path)

        if input_des is None:
            return Response({'error': 'No descriptors found in input image'}, status=400)

        best_match = None
        best_score = 0

        for fingerprint in Fingerprint.objects.all():
            if fingerprint.descriptors is None:
                continue

            db_des = np.frombuffer(fingerprint.descriptors, dtype=np.uint8).reshape(-1, 32)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(input_des, db_des)
            good_matches = [m for m in matches if m.distance < 60]

            score = len(good_matches) / len(input_des) if len(input_des) > 0 else 0

            if score > best_score:
                best_score = score
                best_match = fingerprint.name

        if best_score >= 0.1:
            return Response({
                'match': best_match,
                'score': round(best_score, 2)
            })
        else:
            return Response({'message': 'No match found'}, status=404)
