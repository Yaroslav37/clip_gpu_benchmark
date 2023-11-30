from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('process/', views.process_image, name='process_image'),
    path('search/', views.search, name='search'),
    path('comparison/', views.image_text_comparison, name='image_text_comparison'),
]