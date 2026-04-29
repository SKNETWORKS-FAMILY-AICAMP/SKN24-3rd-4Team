from django.urls import path
from .views import chat_main, send_chat

urlpatterns = [
    path('', chat_main),
    path('send_chat', send_chat, name='send_chat'),
]