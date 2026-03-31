# backend/back/forms.py
from django import forms

class VideoUploadForm(forms.Form):
    video = forms.FileField(
        label='Select a room video',
        required=True,
        widget=forms.ClearableFileInput(attrs={'accept': 'video/*'})
    )