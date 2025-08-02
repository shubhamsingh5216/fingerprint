from django.db import models

class Fingerprint(models.Model):
    name = models.CharField(max_length=100)
    image = models.ImageField(upload_to='fingerprints/')
    descriptors = models.BinaryField(null=True, blank=True)  # ORB features
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
