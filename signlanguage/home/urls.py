"""
URL configuration for signlanguage project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views
from home import views

urlpatterns = [
     path("", views.index, name='home'),
     path('camera-feed/', views.camera_feed, name='camera_feed'),
     path('detect-sign-language/', views.detect_sign_language, name='detect_sign_language'),
     path('animation/',views.animation_view,name='animation'),
     
]



