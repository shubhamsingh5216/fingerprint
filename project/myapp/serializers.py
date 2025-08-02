import base64
import uuid
from django.core.files.base import ContentFile
from rest_framework import serializers
from .models import Fingerprint

class Base64ImageField(serializers.ImageField):
    def to_internal_value(self, data):
        # Check if this is a base64 string
        if isinstance(data, str) and data.startswith('data:image'):
            # Format: data:image/png;base64,xxxxxx
            format, imgstr = data.split(';base64,') 
            ext = format.split('/')[-1] 
            img_name = f"{uuid.uuid4()}.{ext}"
            data = ContentFile(base64.b64decode(imgstr), name=img_name)
        return super().to_internal_value(data)

class FingerprintSerializer(serializers.ModelSerializer):
    image = Base64ImageField()

    class Meta:
        model = Fingerprint
        fields = ['id', 'name', 'image']
