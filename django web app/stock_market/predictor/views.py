from django.shortcuts import render
import json
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from django.contrib.auth.decorators import login_required
from visualizer.views import stocks_daily, forex_daily

# Create your views here.


@login_required
def price_predictor_stocks(request, symbol):
    context = {'symbol': symbol}
    if request.method == 'POST':
        model = request.POST.get('model')

        if model == 'ARIMA':
            context['output'] = ''
        elif model == 'LSTM':

            if os.path.exists('predictor/models/stocks/LSTM/' + context['symbol'] + '_LSTM_predictor'):
                context['model'] = 'LSTM'

                daily_response = stocks_daily(symbol)
                daily_obj = json.loads(daily_response)
                time_series_daily = daily_obj['Time Series (Daily)']

                data_date = list(time_series_daily.keys())[0]

                data_close_price = [float(time_series_daily[date]['4. close']) for date in
                                    list(time_series_daily.keys())[:20]]
                data_close_price.reverse()

                # print(data_date)
                # print(data_close_price)

                context['ltp'] = data_close_price[-1]
                context['ltd'] = data_date

                np_data_close_price = np.array(data_close_price).reshape(-1, 1)
                scalar = MinMaxScaler()
                normalized_data_close_price = scalar.fit_transform(np_data_close_price)

                # print(normalized_data_close_price)
                # print(normalized_data_close_price.shape)

                fin_data = normalized_data_close_price

                # print(x_input)
                # print(x_input.shape)

                price_predictor = load_model('predictor/models/stocks/LSTM/' + context['symbol'] + '_LSTM_predictor')

                for i in range(30):
                    x_input = fin_data[-20:].reshape(1, 20, 1)
                    price = price_predictor.predict(x_input)
                    fin_data = np.append(fin_data, price)

                fin_prd = scalar.inverse_transform(fin_data.reshape(-1, 1))
                fin_prd = np.round_(fin_prd, 2)

                print(fin_prd)

                f = [i[0] for i in fin_prd[20:].tolist()]
                context['output'] = f

                # print(f)
                context['days'] = list(range(1, 31))

            else:
                context['error'] = context['symbol'] + ' LSTM model not found'

        elif model == 'GRU':
            context['output'] = ''

        return render(request, 'predictor/stocks_price_predictor.html', context)

    else:
        return render(request, 'predictor/stocks_price_predictor.html', context)


@login_required
def price_predictor_forex(request, from_symbol, to_symbol):
    context = {'from_symbol': from_symbol, 'to_symbol': to_symbol}

    if request.method == 'POST':
        model = request.POST.get('model')

        if model == 'arima':
            context['output'] = ''
        elif model == 'LSTM':

            if os.path.exists('predictor/models/forex/LSTM/' + context['from_symbol'] + '_' + context['to_symbol']
                              + '_LSTM_predictor'):
                context['model'] = 'LSTM'

                daily_response = forex_daily(from_symbol, to_symbol)
                daily_obj = json.loads(daily_response)
                time_series_daily = daily_obj['Time Series FX (Daily)']

                data_date = list(time_series_daily.keys())[0]

                data_close_price = [float(time_series_daily[date]['4. close']) for date in
                                    list(time_series_daily.keys())[:20]]
                data_close_price.reverse()

                # print(data_date)
                # print(data_close_price)

                context['ltp'] = data_close_price[-1]
                context['ltd'] = data_date

                np_data_close_price = np.array(data_close_price).reshape(-1, 1)
                scalar = MinMaxScaler()
                normalized_data_close_price = scalar.fit_transform(np_data_close_price)

                # print(normalized_data_close_price)
                # print(normalized_data_close_price.shape)

                fin_data = normalized_data_close_price

                # print(x_input)
                # print(x_input.shape)

                price_predictor = load_model('predictor/models/forex/LSTM/' + context['from_symbol'] + '_'
                                             + context['to_symbol'] + '_LSTM_predictor')

                for i in range(30):
                    x_input = fin_data[-20:].reshape(1, 20, 1)
                    price = price_predictor.predict(x_input)
                    fin_data = np.append(fin_data, price)

                fin_prd = scalar.inverse_transform(fin_data.reshape(-1, 1))
                fin_prd = np.round_(fin_prd, 4)

                print(fin_prd)

                f = [i[0] for i in fin_prd[20:].tolist()]
                context['output'] = f

                # print(f)
                context['days'] = list(range(1, 31))

            else:
                context['error'] = context['from_symbol'] + '_' + context['to_symbol'] + ' LSTM model not found'

        elif model == 'gru':
            context['output'] = ''

        return render(request, 'predictor/forex_price_predictor.html', context)

    else:
        return render(request, 'predictor/forex_price_predictor.html', context)
