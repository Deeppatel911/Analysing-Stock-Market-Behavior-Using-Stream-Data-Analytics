from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='market_home'),
    path('reference/<str:exchange>/stocks_list/', views.stocks_list, name='stocks_list'),
    path('reference/symbol_search/', views.symbol_search, name='symbol_search'),
    path('reference/forex_pair_list/', views.forex_pair_list, name='forex_list'),
    path('reference/exchange_list/', views.exchange_list, name='exchange_list'),
    path('core_data/<str:exchange>/<str:symbol>/details/', views.stocks_details, name='stock_details'),
    path('core_data/<str:exchange>/<str:from_symbol>/<str:to_symbol>/details/', views.forex_details,
         name='forex_details'),
    path('core_data/currency_conversion/', views.currency_conversion, name='currency_conversion'),
    path('core_data/exchange_rate/', views.exchange_rate, name='exchange_rate'),
    path('fundamentals/<str:symbol>/cash_flow/', views.cash_flow, name='cash_flow'),
    path('fundamentals/ipo/', views.ipo, name='ipo'),
    path('fundamentals/earnings/', views.earnings, name='earnings'),
    path('fundamentals/<str:symbol>/balance_sheet/', views.balance_sheet, name='balance_sheet'),
    path('fundamentals/<str:symbol>/company_overview/', views.company_overview, name='company_overview'),
    path('fundamentals/market_news/', views.market_news, name='market_news'),
    path('fundamentals/peers/<str:symbol>/', views.peers, name='peers'),
    path('fundamentals/fda_calendar/', views.fda_calendar, name='fda_calendar'),
    path('fundamentals/<str:symbol>/sec_filings/', views.sec_filings, name='sec_filings'),
    path('fundamentals/<str:symbol>/company_news/', views.company_news, name='company_news'),
    path('fundamentals/<str:symbol>/uspto_patents/', views.uspto_patents, name='uspto_patents')

]
