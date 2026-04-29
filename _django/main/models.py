from django.db import models
from django import forms

class Chat(models.Model):
    type = models.CharField(max_length=200, default='user')
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

class ChatForm(forms.ModelForm):
    class Meta:
        model = Chat
        fields = ['content']