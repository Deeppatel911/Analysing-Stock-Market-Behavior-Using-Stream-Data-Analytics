from django.urls import path
from . import views

urlpatterns = [
    path('price_predictor/stocks/<str:symbol>/', views.price_predictor_stocks, name='price_predictor_stocks'),
    path('price_predictor/forex/<str:from_symbol>/<str:to_symbol>/', views.price_predictor_forex,
         name='price_predictor_forex')

]
