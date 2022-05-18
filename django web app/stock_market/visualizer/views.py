from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
from market.views import *

# Create your views here.


# technical indicators
def adx(symbol, interval):
    adx_url = twelveDataUrl + 'adx'
    adx_query = {'symbol': symbol, 'interval': interval,
                 'apikey': twelveDataApiKey}
    adx_response = requests.request('GET', adx_url, params=adx_query)
    return adx_response.text


# technical indicators
def bbands(symbol, interval):
    bbands_url = twelveDataUrl + 'bbands'
    bbands_query = {'symbol': symbol,
                    'interval': interval, 'apikey': twelveDataApiKey}
    bbands_response = requests.request('GET', bbands_url, params=bbands_query)
    return bbands_response.text


# technical indicators
def ema(symbol, interval):
    ema_url = twelveDataUrl + 'ema'
    ema_query = {'symbol': symbol, 'interval': interval,
                 'apikey': twelveDataApiKey}
    ema_response = requests.request('GET', ema_url, params=ema_query)
    return ema_response.text


# technical indicators
def macd(symbol, interval):
    macd_url = twelveDataUrl + 'macd'
    macd_query = {'symbol': symbol, 'interval': interval, 'apikey': twelveDataApiKey}
    macd_response = requests.request('GET', macd_url, params=macd_query)
    return macd_response.text


# technical indicators
def percent_b(symbol, interval):
    percent_b_url = twelveDataUrl + 'percent_b'
    percent_b_query = {'symbol': symbol, 'interval': interval, 'apikey': twelveDataApiKey}
    percent_b_response = requests.request('GET', percent_b_url, params=percent_b_query)
    return percent_b_response.text


# technical indicators
def rsi(symbol, interval):
    rsi_url = twelveDataUrl + 'rsi'
    rsi_query = {'symbol': symbol, 'interval': interval, 'apikey': twelveDataApiKey}
    rsi_response = requests.request('GET', rsi_url, params=rsi_query)
    return rsi_response.text


# technical indicators
def stoch(symbol, interval):
    stoch_url = twelveDataUrl + 'stoch'
    stoch_query = {'symbol': symbol, 'interval': interval, 'apikey': twelveDataApiKey}
    stoch_response = requests.request('GET', stoch_url, params=stoch_query)
    return stoch_response.text


# technical indicators
def sma(symbol, interval):
    sma_url = twelveDataUrl + 'sma'
    sma_query = {'symbol': symbol, 'interval': interval, 'apikey': twelveDataApiKey}
    sma_response = requests.request('GET', sma_url, params=sma_query)
    return sma_response.text


# technical indicators
def vwap(symbol, interval):
    vwap_url = twelveDataUrl + 'vwap'
    vwap_query = {'symbol': symbol, 'interval': interval, 'apikey': twelveDataApiKey}
    vwap_response = requests.request('GET', vwap_url, params=vwap_query)
    return vwap_response.text


# stocks time series
def stocks_intra_day(symbol, interval):
    stocks_intra_day_url = alphaUrl + 'query'
    stocks_intra_day_query = {'apikey': alphaApiKey, 'interval': interval, 'function': 'TIME_SERIES_INTRADAY',
                              'symbol': symbol}
    stocks_intra_day_response = requests.request('GET', stocks_intra_day_url, params=stocks_intra_day_query)
    return stocks_intra_day_response.text


# stocks time series
def stocks_daily(symbol):
    stocks_daily_url = alphaUrl + 'query'
    stocks_daily_query = {'apikey': alphaApiKey, 'function': 'TIME_SERIES_DAILY', 'symbol': symbol}
    stocks_daily_response = requests.request('GET', stocks_daily_url, params=stocks_daily_query)
    return stocks_daily_response.text


# stocks time series
def stocks_weekly(symbol):
    stocks_weekly_url = alphaUrl + 'query'
    stocks_weekly_query = {'apikey': alphaApiKey, 'function': 'TIME_SERIES_WEEKLY', 'symbol': symbol}
    stocks_weekly_response = requests.request('GET', stocks_weekly_url, params=stocks_weekly_query)
    return stocks_weekly_response.text


# stocks time series
def stocks_monthly(symbol):
    stocks_monthly_url = alphaUrl + 'query'
    stocks_monthly_query = {'apikey': alphaApiKey, 'function': 'TIME_SERIES_MONTHLY', 'symbol': symbol}
    stocks_monthly_response = requests.request('GET', stocks_monthly_url, params=stocks_monthly_query)
    return stocks_monthly_response.text


# forex time series
def forex_intraday(from_symbol, to_symbol, interval):
    forex_intra_day_url = alphaUrl + 'query'
    forex_intra_day_query = {'apikey': alphaApiKey, 'interval': interval, 'function': 'FX_INTRADAY',
                             'from_symbol': from_symbol, 'to_symbol': to_symbol}
    forex_intra_day_response = requests.request('GET', forex_intra_day_url, params=forex_intra_day_query)
    return forex_intra_day_response.text


# forex time series
def forex_daily(from_symbol, to_symbol):
    forex_daily_url = alphaUrl + 'query'
    forex_daily_query = {'apikey': alphaApiKey, 'function': 'FX_DAILY', 'from_symbol': from_symbol,
                         'to_symbol': to_symbol}
    forex_daily_response = requests.request('GET', forex_daily_url, params=forex_daily_query)
    return forex_daily_response.text


# forex time series
def forex_weekly(from_symbol, to_symbol):
    forex_weekly_url = alphaUrl + 'query'
    forex_weekly_query = {'apikey': alphaApiKey, 'function': 'FX_WEEKLY', 'from_symbol': from_symbol,
                          'to_symbol': to_symbol}
    forex_weekly_response = requests.request('GET', forex_weekly_url, params=forex_weekly_query)
    return forex_weekly_response.text


# forex time series
def forex_monthly(from_symbol, to_symbol):
    forex_monthly_url = alphaUrl + 'query'
    forex_monthly_query = {'apikey': alphaApiKey, 'function': 'FX_MONTHLY', 'from_symbol': from_symbol,
                           'to_symbol': to_symbol}
    forex_monthly_response = requests.request('GET', forex_monthly_url, params=forex_monthly_query)
    return forex_monthly_response.text


@login_required
def technical_indicators_stocks(request, symbol):
    return render(request, 'visualizer/technical_indicators.html', {'type': 'Stock', 'symbol': symbol})


@login_required
def technical_indicators_forex(request, from_symbol, to_symbol):
    return render(request, 'visualizer/technical_indicators.html', {'type': 'Forex',
                                                                    'symbol': from_symbol + '/' + to_symbol})


@csrf_exempt
def get_technical_indicator_data(request):
    if request.is_ajax():
        selected_value = request.POST.get('indicator')
        interval = request.POST.get('interval')
        symbol = request.POST.get('symbol')
        context = {'symbol': symbol}

        if selected_value == 'adx':
            adx_op = adx(symbol, interval)
            context['output'] = json.loads(adx_op)

        elif selected_value == 'bbands':
            bbands_op = bbands(symbol, interval)
            context['output'] = json.loads(bbands_op)

        elif selected_value == 'ema':
            ema_op = ema(symbol, interval)
            context['output'] = json.loads(ema_op)

        elif selected_value == 'macd':
            macd_op = macd(symbol, interval)
            context['output'] = json.loads(macd_op)

        elif selected_value == 'percent_b':
            percent_b_op = percent_b(symbol, interval)
            context['output'] = json.loads(percent_b_op)

        elif selected_value == 'rsi':
            rsi_op = rsi(symbol, interval)
            context['output'] = json.loads(rsi_op)

        elif selected_value == 'stoch':
            stoch_op = stoch(symbol, interval)
            context['output'] = json.loads(stoch_op)

        elif selected_value == 'sma':
            sma_op = sma(symbol, interval)
            context['output'] = json.loads(sma_op)

        elif selected_value == 'vwap':
            vwap_op = vwap(symbol, interval)
            context['output'] = json.loads(vwap_op)

        # print(context)

        return JsonResponse(context)


@login_required
def time_series_stocks(request, symbol):
    return render(request, 'visualizer/time_series_stocks.html', {'type': 'Stock', 'symbol': symbol})


@csrf_exempt
def get_time_series_data_stock(request):
    if request.is_ajax():
        selected_value = request.POST.get('time_series')
        symbol = request.POST.get('symbol')
        context = {'symbol': symbol}

        if selected_value == 'intraday':
            interval = request.POST.get('interval')
            intraday_op = stocks_intra_day(symbol, interval)
            context['output'] = json.loads(intraday_op)

        elif selected_value == 'daily':
            daily_op = stocks_daily(symbol)
            context['output'] = json.loads(daily_op)

        elif selected_value == 'weekly':
            weekly_op = stocks_weekly(symbol)
            context['output'] = json.loads(weekly_op)

        elif selected_value == 'monthly':
            monthly_op = stocks_monthly(symbol)
            context['output'] = json.loads(monthly_op)

        # print(context)

        return JsonResponse(context)


@login_required
def time_series_forex(request, from_symbol, to_symbol):
    return render(request, 'visualizer/time_series_forex.html', {'type': 'Forex', 'from_symbol': from_symbol,
                                                                 'to_symbol': to_symbol})


@csrf_exempt
def get_time_series_data_forex(request):
    if request.is_ajax():
        selected_value = request.POST.get('time_series')
        from_symbol = str(request.POST.get('from_symbol'))
        to_symbol = str(request.POST.get('to_symbol'))
        context = {}

        if selected_value == 'intraday':
            interval = request.POST.get('interval')
            intraday_op = forex_intraday(from_symbol, to_symbol, interval)
            context['output'] = json.loads(intraday_op)

        if selected_value == 'daily':
            daily_op = forex_daily(from_symbol, to_symbol)
            context['output'] = json.loads(daily_op)

        if selected_value == 'weekly':
            weekly_op = forex_weekly(from_symbol, to_symbol)
            context['output'] = json.loads(weekly_op)

        if selected_value == 'monthly':
            monthly_op = forex_monthly(from_symbol, to_symbol)
            context['output'] = json.loads(monthly_op)

        # print(context)

        return JsonResponse(context)


@login_required
def social_sentiment(request, symbol):
    return render(request, 'visualizer/social_sentiment.html', {'symbol': symbol})


@csrf_exempt
def get_social_sentiment_data(request):
    if request.is_ajax():
        symbol = request.POST.get('symbol')
        context = {'symbol': symbol}

        from_date = request.POST.get('from_date')
        to_date = request.POST.get('to_date')

        social_sentiment_url = finnhubUrl + '/stock/social-sentiment'

        if not from_date:
            # print(from_date)
            query_string = {'symbol': symbol, 'token': finnhubApiKey}
        else:
            query_string = {'symbol': symbol, 'token': finnhubApiKey, 'from': from_date, 'to': to_date}

        social_sentiment_response = requests.request('GET', social_sentiment_url, params=query_string)

        context['output'] = json.loads(social_sentiment_response.text)

        # print(context)

        return JsonResponse(context)


# economic indicators
@login_required
def economic_indicators(request, indicator):
    return render(request, 'visualizer/economic_indicators.html', {'indicator': indicator})


@csrf_exempt
def get_economic_indicator_data(request):
    if request.is_ajax():
        indicator = request.POST.get('indicator')
        context = {}

        if indicator == 'gdp':
            gdp_url = alphaUrl + 'query'
            gdp_query = {'apikey': alphaApiKey, 'function': 'REAL_GDP'}
            gdp_response = requests.request('GET', gdp_url, params=gdp_query)
            context['output'] = json.loads(gdp_response.text)

        if indicator == 'cpi':
            cpi_url = alphaUrl + 'query'
            cpi_query = {'apikey': alphaApiKey, 'function': 'CPI', 'interval': 'semiannual'}
            cpi_response = requests.request('GET', cpi_url, params=cpi_query)
            context['output'] = json.loads(cpi_response.text)

        if indicator == 'treasury_yield':
            treasure_yield_url = alphaUrl + 'query'
            treasure_yield_query = {'apikey': alphaApiKey, 'function': 'TREASURY_YIELD'}
            treasure_yield_response = requests.request('GET', treasure_yield_url, params=treasure_yield_query)
            context['output'] = json.loads(treasure_yield_response.text)

        if indicator == 'federal_funds_rate':
            federal_funds_url = alphaUrl + 'query'
            federal_funds_query = {'apikey': alphaApiKey, 'function': 'FEDERAL_FUNDS_RATE'}
            federal_funds_response = requests.request('GET', federal_funds_url, params=federal_funds_query)
            context['output'] = json.loads(federal_funds_response.text)

        if indicator == 'unemployment_rate':
            unemployment_rate_url = alphaUrl + 'query'
            unemployment_rate_query = {'apikey': alphaApiKey, 'function': 'UNEMPLOYMENT'}
            unemployment_rate_response = requests.request('GET', unemployment_rate_url, params=unemployment_rate_query)
            context['output'] = json.loads(unemployment_rate_response.text)

        if indicator == 'inflation_expectation':
            inflation_exception_url = alphaUrl + 'query'
            inflation_exception_query = {'apikey': alphaApiKey, 'function': 'INFLATION_EXPECTATION'}
            inflation_exception_response = requests.request('GET', inflation_exception_url,
                                                            params=inflation_exception_query)
            context['output'] = json.loads(inflation_exception_response.text)

        # print(context['output']['name'])
        # for v in context['output']['data']:
        #    print(v['date'])
        #    print(v['value'])

        return JsonResponse(context)


# fundamentals
@login_required
def income_statement(request, symbol):
    return render(request, 'visualizer/fundamentals_income_statement.html', {'symbol': symbol})


@csrf_exempt
def get_income_statement(request):
    if request.is_ajax():
        symbol = request.POST.get('symbol')
        context = {'symbol': symbol}

        income_statement_url = alphaUrl + 'query'
        income_statement_query = {'apikey': alphaApiKey, 'function': 'INCOME_STATEMENT', 'symbol': symbol}
        income_statement_response = requests.request('GET', income_statement_url, params=income_statement_query)
        context['output'] = json.loads(income_statement_response.text)

        return JsonResponse(context)
