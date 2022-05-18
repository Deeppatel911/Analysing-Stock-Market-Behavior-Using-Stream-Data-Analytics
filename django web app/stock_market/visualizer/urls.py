from django.urls import path
from . import views

urlpatterns = [
    path('technical_indicators/stocks/<str:symbol>/', views.technical_indicators_stocks,
         name='technical_indicators_stocks'),
    path('technical_indicators/forex/<str:from_symbol>/<str:to_symbol>/', views.technical_indicators_forex,
         name='technical_indicators_forex'),
    path('get_technical_indicator_data/', views.get_technical_indicator_data, name='get_technical_indicator_data'),
    path('time_series/stocks/<str:symbol>/', views.time_series_stocks, name='time_series_stocks'),
    path('get_time_series_data/stock/', views.get_time_series_data_stock, name='get_time_series_data_stock'),
    path('time_series/forex/<str:from_symbol>/<str:to_symbol>/', views.time_series_forex, name='time_series_forex'),
    path('get_time_series_data/forex/', views.get_time_series_data_forex, name='time_series_data_forex'),
    path('social_sentiment/<str:symbol>/', views.social_sentiment, name='social_sentiment'),
    path('get_social_sentiment_data/', views.get_social_sentiment_data, name='get_social_sentiment_data'),
    path('economic_indicators/<str:indicator>/', views.economic_indicators, name='economic_indicators'),
    path('get_economic_indicator_data/', views.get_economic_indicator_data, name='get_economic_indicator_data'),
    path('fundamentals/<str:symbol>/income_statement/', views.income_statement, name='income_statement'),
    path('fundamentals/get_income_statement/', views.get_income_statement, name='get_income_statement')

]
