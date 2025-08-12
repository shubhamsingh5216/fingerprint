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
    # Increase keypoints and make it scale invariant
    orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2)
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

        # Extract fingerprint descriptors
        input_des = extract_orb_features(image_path)
        print(f"Input descriptors shape: {input_des.shape if input_des is not None else 'None'}")

        if input_des is None:
            return Response({'error': 'Could not extract features from image'}, status=400)

        # Ensure dtype and save
        input_des = input_des.astype(np.uint8)

        fingerprint = Fingerprint(name=name)
        fingerprint.image.save(os.path.basename(image_path), open(image_path, 'rb'))
        fingerprint.descriptors = input_des.tobytes()
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

        input_des = input_des.astype(np.uint8)

        best_match = None
        best_score = 0

        for fingerprint in Fingerprint.objects.all():
            if fingerprint.descriptors is None:
                continue

            try:
                db_des = np.frombuffer(fingerprint.descriptors, dtype=np.uint8).reshape(-1, 32)
            except ValueError:
                continue  # Skip corrupted descriptor data

            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(input_des, db_des, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            score = len(good_matches) / len(input_des)
            print(f"Comparing with {fingerprint.name}: score = {score}")

            if score < best_score:
                best_score = score
                best_match = fingerprint.name

        if best_score >= 0.5:
            return Response({
                'match': best_match,
                'score': round(best_score, 2)
            })
        else:
            print(f"Comparing with {fingerprint.name}: score = {score}")
            return Response({
            'message': 'No match found',
            'score': score
        }, status=404)

            
