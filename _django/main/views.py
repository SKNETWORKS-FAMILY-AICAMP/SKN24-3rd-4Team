from django.shortcuts import render
from .models import Chat, ChatForm
from django.shortcuts import redirect

import requests

from django.conf import settings

def chat_main(request):
    form = ChatForm(request.POST or None)
    chats = Chat.objects.all().order_by('created_at')
    return render(request, 'main/chat_main.html', {'form': form, 'chats': chats})

def send_chat(request):
    form = ChatForm(request.POST or None)
    if form.is_valid():
        form.save()
        # 채팅 저장 후 FAST API 호출하여 답변 받아오기
        send_url = settings.FAST_API_URL + '/chat'
        # FAST API로 채팅 내용 전송
        chat_data = {
            "user_id": "user_1",
            "session_id": "session_1",
            "insurer": "cigna",
            "message": form.cleaned_data['content'],
            "chat_history": []
        }
        response = requests.post(send_url, json=chat_data)
        print("🚀 FAST API 응답:", response.json())
        if response.status_code == 200:
            answer = response.json().get('messages', [])[-1].get('content', '')
            # 답변을 Chat 모델에 저장
            Chat.objects.create(type='bot', content=answer)
        chats = Chat.objects.all().order_by('created_at')
        return render(request, 'main/chat_main.html', {'form': form, 'chats': chats})
    chats = Chat.objects.all().order_by('created_at')
    return redirect('main:chat_main')