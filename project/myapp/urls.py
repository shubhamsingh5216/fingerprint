from django.urls import path
from .views import AddFingerprintView, MatchFingerprintView
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('add/', AddFingerprintView.as_view(), name='add-fingerprint'),
    path('match/', MatchFingerprintView.as_view(), name='match-fingerprint'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)