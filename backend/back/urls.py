# back/urls.py
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from back.views import RunFullPipelineAPIView

urlpatterns = [
    path('run-full-pipeline/', RunFullPipelineAPIView.as_view(), name='run_full_pipeline'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)