from django.urls import path
from . import views

app_name = 'chat' # Namespace para las URLs de esta app

urlpatterns = [
    path('', views.chat_view, name='chat_view'),        # Vista principal del chat
    path('api/chat/', views.chat_api, name='chat_api'), # Endpoint de la API del chat
]