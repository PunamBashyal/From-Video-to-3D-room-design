from django.db import models

# Create your database models here (if needed)

class VideoUpload(models.Model):
    video = models.FileField(upload_to='videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)