from django.urls import path
from . import views

urlpatterns = [
    path('', views.scrape_products, name='scrape_products'),
]
