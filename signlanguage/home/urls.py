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
     path('', views.index, name='home'),
    #  path('video/', views.video_feed, name='video'),
     path('camera-feed/', views.camera_feed, name='camera_feed'),
     path('camera-view/', views.camera_view, name='camera_view'),
     path('perform-prediction/', views.perform_prediction, name='perform_prediction'),

    #  path('prediction/', views.perform_prediction, name='perform_prediction'),


     path('ncamera-feed/', views.ncamera_feed, name='ncamera_feed'),
     path('ncamera-view/', views.ncamera_view, name='ncamera_feed'),


 
    #  path('detect-sign-language/', views.detect_sign_language, name='detect_sign_language'),
     path('animation/',views.animation_view,name='animation'),
     path('nanimation/',views.nanimation_view,name='nanimation'),

    #    path('video_feed', views.video_feed, name='video_feed'),
    # path('webcam_feed', views.webcam_feed, name='webcam_feed'),
    # path('mask_feed', views.mask_feed, name='mask_feed'),
	# path('livecam_feed', views.livecam_feed, name='livecam_feed'),
     
]



