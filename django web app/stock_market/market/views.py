from django.shortcuts import render
import requests
import json
import csv
import time
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator

# Create your views here.

twelveDataUrl = 'https://api.twelvedata.com/'
twelveDataApiKey = 'YOUR_API_KEY'
alphaUrl = 'https://www.alphavantage.co/'
alphaApiKey = 'YOUR_API_KEY'
finnhubUrl = 'https://finnhub.io/api/v1'
finnhubApiKey = 'YOUR_API_KEY'


# core data
def real_time_price(symbol, exchange):
    price_url = twelveDataUrl + 'price'
    price_query = {'symbol': symbol, 'exchange': exchange, 'apikey': twelveDataApiKey}
    price_response = requests.request('GET', price_url, params=price_query)
    return price_response.text


# core data
def quote_endpoint(symbol, exchange):
    quote_url = twelveDataUrl + 'quote'
    quote_query = {'symbol': symbol, 'exchange': exchange, 'apikey': twelveDataApiKey}
    quote_response = requests.request('GET', quote_url, params=quote_query)
    return quote_response.text


# fundamental data
def logo(symbol):
    logo_url = twelveDataUrl + 'logo'
    logo_query = {'symbol': symbol, 'apikey': twelveDataApiKey}
    logo_response = requests.request('GET', logo_url, params=logo_query)
    return logo_response.text


def home(request):
    return render(request, 'market/home.html')


# reference data
@login_required
def stocks_list(request, exchange):
    # if request.method == 'POST':
    #    exchange = str(request.POST.get('exchange'))
    #    exchange = exchange.upper()

    query_string = {'exchange': exchange, 'format': 'CSV'}
    stock_url = twelveDataUrl + 'stocks'
    stocks_response = requests.request('GET', stock_url, params=query_string)
    # stocks_output = json.loads(stocks_response.text)
    # stocks_output2 = stocks_output['data']

    stocks_obj = stocks_response.text
    op = csv.reader(stocks_obj.splitlines(), delimiter=';')
    final_op = list(op)

    paginator = Paginator(final_op[1:], 5)
    page_no = request.GET.get('page')
    page_obj = paginator.get_page(page_no)

    return render(request, 'market/reference_stocks_list.html', {'stocks_list': page_obj})

    # else:
    #    return render(request, 'market/reference_stocks_list.html')


# reference data
@login_required
def symbol_search(request):
    if request.method == 'POST':
        symbol = str(request.POST.get('symbol'))
        symbol = symbol.upper()

        query_string = {'symbol': symbol}
        symbol_search_url = twelveDataUrl + 'symbol_search'
        symbol_search_response = requests.request('GET', symbol_search_url, params=query_string)
        stocks_output = json.loads(symbol_search_response.text)
        stocks_output2 = stocks_output['data']

        # paginator = Paginator(stocks_output2, 10)
        # page_no = request.GET.get('page')
        # page_obj = paginator.get_page(page_no)

        context = {'symbol_search': stocks_output2}
        from_symbol = symbol[:3]
        to_symbol = symbol[4:]
        context['from_symbol'] = from_symbol
        context['to_symbol'] = to_symbol

        return render(request, 'market/reference_symbol_search.html', context)


# reference data
@login_required
def forex_pair_list(request):
    forex_pair_url = twelveDataUrl + 'forex_pairs'
    forex_pair_query = {'format': 'CSV'}
    forex_pair_response = requests.request('GET', forex_pair_url, params=forex_pair_query)
    # forex_pair_output = json.loads(forex_pair_response.text)

    stocks_obj = forex_pair_response.text
    op = csv.reader(stocks_obj.splitlines(), delimiter=';')
    final_op = list(op)

    paginator = Paginator(final_op[1:], 6)
    page_no = request.GET.get('page')
    page_obj = paginator.get_page(page_no)

    return render(request, 'market/reference_forexpair_list.html', {'forex_pairs': page_obj})


# reference data
@login_required
def exchange_list(request):
    exchanges_url = twelveDataUrl + 'exchanges'
    exchanges_query = {'format': 'CSV'}
    exchanges_response = requests.request('GET', exchanges_url, params=exchanges_query)
    # exchanges_output = json.loads(exchanges_response.text)

    # exchanges_obj = exchanges_response.text
    exchanges_obj = exchanges_response.text
    op = csv.reader(exchanges_obj.splitlines(), delimiter=';')
    final_op = list(op)

    paginator = Paginator(final_op[1:], 6)
    page_no = request.GET.get('page')
    page_obj = paginator.get_page(page_no)

    return render(request, 'market/reference_exchange_list.html', {'exchanges': page_obj})


# core data
@login_required
def stocks_details(request, symbol, exchange):
    symbol = (str(symbol)).upper()
    exchange = (str(exchange)).upper()
    context = {}

    if exchange == 'NASDAQ' or exchange == 'NYSE':
        price = real_time_price(symbol, exchange)
        price_obj = json.loads(price)
        quote = quote_endpoint(symbol, exchange)
        quote_obj = json.loads(quote)
        company_logo = logo(symbol)
        company_logo_obj = json.loads(company_logo)

        quote_obj['open'] = round(float(quote_obj['open']), 2)
        quote_obj['high'] = round(float(quote_obj['high']), 2)
        quote_obj['low'] = round(float(quote_obj['low']), 2)
        quote_obj['close'] = round(float(quote_obj['close']), 2)
        quote_obj['previous_close'] = round(float(quote_obj['previous_close']), 2)
        quote_obj['change'] = round(float(quote_obj['change']), 2)
        quote_obj['percent_change'] = round(float(quote_obj['percent_change']), 2)

        quote_obj['fifty_two_week']['high'] = round(float(quote_obj['fifty_two_week']['high']), 2)
        quote_obj['fifty_two_week']['low'] = round(float(quote_obj['fifty_two_week']['low']), 2)
        quote_obj['fifty_two_week']['low_change'] = round(float(quote_obj['fifty_two_week']['low_change']), 2)
        quote_obj['fifty_two_week']['high_change'] = round(float(quote_obj['fifty_two_week']['high_change']), 2)
        quote_obj['fifty_two_week']['low_change_percent'] = round(
                                                        float(quote_obj['fifty_two_week']['low_change_percent']), 2)
        quote_obj['fifty_two_week']['high_change_percent'] = round(
                                                        float(quote_obj['fifty_two_week']['high_change_percent']), 2)

        context['price'] = round(float(price_obj['price']), 2)
        context['quote'] = quote_obj
        context['logo'] = company_logo_obj

        return render(request, 'market/core_data_stock_details.html', context)

    else:
        return render(request, 'market/reference_stocks_list.html', {'error': '404 Error'})


@login_required
def forex_details(request, exchange, from_symbol, to_symbol):
    context = {}

    price = real_time_price(from_symbol + '/' + to_symbol, exchange)
    price_obj = json.loads(price)
    quote = quote_endpoint(from_symbol + '/' + to_symbol, exchange)
    quote_obj = json.loads(quote)
    company_logo = logo(from_symbol + '/' + to_symbol)
    company_logo_obj = json.loads(company_logo)

    context['from_symbol'] = from_symbol
    context['to_symbol'] = to_symbol
    context['exchange'] = exchange
    context['symbol'] = from_symbol + '/' + to_symbol
    context['price'] = round(float(price_obj['price']), 2)

    quote_obj['open'] = round(float(quote_obj['open']), 2)
    quote_obj['high'] = round(float(quote_obj['high']), 2)
    quote_obj['low'] = round(float(quote_obj['low']), 2)
    quote_obj['close'] = round(float(quote_obj['close']), 2)
    quote_obj['previous_close'] = round(float(quote_obj['previous_close']), 2)
    quote_obj['change'] = round(float(quote_obj['change']), 2)
    quote_obj['percent_change'] = round(float(quote_obj['percent_change']), 2)

    quote_obj['fifty_two_week']['high'] = round(float(quote_obj['fifty_two_week']['high']), 2)
    quote_obj['fifty_two_week']['low'] = round(float(quote_obj['fifty_two_week']['low']), 2)
    quote_obj['fifty_two_week']['low_change'] = round(float(quote_obj['fifty_two_week']['low_change']), 2)
    quote_obj['fifty_two_week']['high_change'] = round(float(quote_obj['fifty_two_week']['high_change']), 2)
    quote_obj['fifty_two_week']['low_change_percent'] = round(
                                                        float(quote_obj['fifty_two_week']['low_change_percent']), 2)
    quote_obj['fifty_two_week']['high_change_percent'] = round(
                                                        float(quote_obj['fifty_two_week']['high_change_percent']), 2)

    context['quote'] = quote_obj
    context['logo'] = company_logo_obj

    return render(request, 'market/core_data_forex_details.html', context)


# core data
@login_required
def exchange_rate(request):
    if request.method == 'POST':
        symbol = str(request.POST.get('symbol'))

        exchange_rate_url = twelveDataUrl + 'exchange_rate'
        exchange_rate_query = {'symbol': symbol, 'apikey': twelveDataApiKey}
        exchange_rate_response = requests.request('GET', exchange_rate_url, params=exchange_rate_query)
        exchange_rate_obj = json.loads(exchange_rate_response.text)
        return render(request, 'market/core_data_exchange_rate.html', {'output': exchange_rate_obj})
    else:
        return render(request, 'market/core_data_exchange_rate.html')


# core data
@login_required
def currency_conversion(request):
    if request.method == 'POST':
        symbol = str(request.POST.get('symbol'))
        amount = str(request.POST.get('amount'))

        currency_url = twelveDataUrl + 'currency_conversion'
        currency_query = {'symbol': symbol, 'amount': amount, 'apikey': twelveDataApiKey}
        currency_response = requests.request('GET', currency_url, params=currency_query)
        currency_obj = json.loads(currency_response.text)
        return render(request, 'market/core_data_currency_conversion.html', {'output': currency_obj})
    else:
        return render(request, 'market/core_data_currency_conversion.html')


# fundamentals
@login_required
def ipo(request):
    ipo_calendar_url = alphaUrl + 'query'
    ipo_calendar_query = {'apikey': alphaApiKey, 'function': 'IPO_CALENDAR'}
    ipo_calendar_response = requests.request('GET', ipo_calendar_url, params=ipo_calendar_query)

    ipo_obj = ipo_calendar_response.text
    # print(ipo_obj)
    # print(ipo_obj.splitlines())
    op = csv.reader(ipo_obj.splitlines(), delimiter=',')
    # print(op)
    final_op = list(op)
    print(final_op)

    if len(final_op) > 1:
        paginator = Paginator(final_op[1:], 5)
        page_no = request.GET.get('page')
        page_obj = paginator.get_page(page_no)

        return render(request, 'market/fundamentals_ipo.html', {'ipo': page_obj})

    else:
        return render(request, 'market/fundamentals_ipo.html', {'error': 'No IPOs expected in the next 3 months.'})


# fundamental data
@login_required
def cash_flow(request, symbol):
    cash_flow_url = alphaUrl + 'query'
    cash_flow_query = {'apikey': alphaApiKey, 'function': 'CASH_FLOW', 'symbol': symbol}
    cash_flow_response = requests.request('GET', cash_flow_url, params=cash_flow_query)
    cash_flow_obj = json.loads(cash_flow_response.text)
    return render(request, 'market/fundamentals_cash_flow.html', {'output': cash_flow_obj})


# fundamental data
@login_required
def earnings(request):
    earnings_calendar_url = alphaUrl + 'query'
    earnings_calendar_query = {'apikey': alphaApiKey, 'function': 'EARNINGS_CALENDAR'}
    earnings_calendar_response = requests.request('GET', earnings_calendar_url, params=earnings_calendar_query)

    earnings_obj = earnings_calendar_response.text
    # print(earnings_obj)
    # print(earnings_obj.splitlines())
    op = csv.reader(earnings_obj.splitlines(), delimiter=',')
    final_op = list(op)

    paginator = Paginator(final_op[1:], 4)
    page_no = request.GET.get('page')
    page_obj = paginator.get_page(page_no)

    return render(request, 'market/fundamentals_earnings.html', {'earnings': page_obj})


# fundamental data
@login_required
def balance_sheet(request, symbol):
    balance_sheet_url = alphaUrl + 'query'
    balance_sheet_query = {'apikey': alphaApiKey, 'function': 'BALANCE_SHEET', 'symbol': symbol}
    balance_sheet_response = requests.request('GET', balance_sheet_url, params=balance_sheet_query)
    balance_sheet_obj = json.loads(balance_sheet_response.text)
    return render(request, 'market/fundamentals_balance_sheet.html', {'output': balance_sheet_obj})


# fundamental data
@login_required
def company_overview(request, symbol):
    company_overview_url = alphaUrl + 'query'
    company_overview_query = {'apikey': alphaApiKey, 'function': 'OVERVIEW', 'symbol': symbol}
    company_overview_response = requests.request('GET', company_overview_url, params=company_overview_query)

    # company_overview_obj = company_overview_response.text
    # op = csv.reader(company_overview_obj.splitlines(), delimiter=',')

    company_overview_obj = company_overview_response.text

    return render(request, 'market/fundamentals_company_overview.html', {'output': company_overview_obj})


# fundamentals (finnhub)
@login_required
def market_news(request):
    context = {}
    if request.method == 'POST':
        category = request.POST.get('news_category')

        market_news_url = finnhubUrl + '/news'
        query_string = {'category': category, 'token': finnhubApiKey}
        market_news_response = requests.request('GET', market_news_url, params=query_string)
        context['output'] = json.loads(market_news_response.text)

        for v in context['output']:
            v['datetime'] = time.asctime(time.localtime(v['datetime']))

        return render(request, 'market/fundamentals_market_news.html', context)

    else:
        return render(request, 'market/fundamentals_market_news.html')


# fundamentals (finnhub)
@login_required
def peers(request, symbol):
    context = {}
    peers_url = finnhubUrl + '/stock/peers'
    peers_string = {'symbol': symbol, 'token': finnhubApiKey}
    peers_response = requests.request('GET', peers_url, params=peers_string)
    context['output'] = json.loads(peers_response.text)

    return render(request, 'market/fundamentals_peers.html', context)


# fundamentals (finnhub)
@login_required
def fda_calendar(request):
    context = {}
    fda_calendar_url = finnhubUrl + '/fda-advisory-committee-calendar'
    fda_calendar_string = {'token': finnhubApiKey}
    fda_calendar_response = requests.request('GET', fda_calendar_url, params=fda_calendar_string)
    context['output'] = json.loads(fda_calendar_response.text)

    return render(request, 'market/fundamentals_fda_calendar.html', context)


# fundamentals (finnhub)
@login_required
def sec_filings(request, symbol):
    context = {}
    sec_filings_url = finnhubUrl + '/stock/filings'
    sec_filings_string = {'symbol': symbol, 'token': finnhubApiKey}
    sec_filings_response = requests.request('GET', sec_filings_url, params=sec_filings_string)
    context['output'] = json.loads(sec_filings_response.text)

    return render(request, 'market/fundamentals_sec_filings.html', context)


# fundamentals (finnhub)
@login_required
def company_news(request, symbol):
    if request.method == 'POST':
        from_date = str(request.POST.get('_from'))
        to_date = str(request.POST.get('_to'))
        context = {'symbol': symbol}

        company_news_url = finnhubUrl + '/company-news'
        company_news_string = {'symbol': symbol, 'from': from_date, 'to': to_date, 'token': finnhubApiKey}
        company_news_response = requests.request('GET', company_news_url, params=company_news_string)
        context['output'] = json.loads(company_news_response.text)

        for v in context['output']:
            v['datetime'] = time.asctime(time.localtime(v['datetime']))

        return render(request, 'market/fundamentals_company_news.html', context)

    else:
        return render(request, 'market/fundamentals_company_news.html', {'symbol': symbol})


# fundamentals (finnhub)
@login_required
def uspto_patents(request, symbol):
    if request.method == 'POST':
        from_date = str(request.POST.get('_from'))
        to_date = str(request.POST.get('_to'))
        context = {'symbol': symbol}

        uspto_patents_url = finnhubUrl + '/stock/uspto-patent'
        uspto_patents_string = {'symbol': symbol, 'from': from_date, 'to': to_date, 'token': finnhubApiKey}
        uspto_patents_response = requests.request('GET', uspto_patents_url, params=uspto_patents_string)
        context['output'] = json.loads(uspto_patents_response.text)

        return render(request, 'market/fundamentals_uspto_patents.html', context)

    else:
        return render(request, 'market/fundamentals_uspto_patents.html', {'symbol': symbol})
