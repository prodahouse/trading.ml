"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

StrategyLearner
Carlos Aguayo
carlos.aguayo@gatech.edu
gtid 903055858

"""

import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut
import numpy as np

import warnings  # http://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action="ignore", category=FutureWarning)


class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.learner = None

    @staticmethod
    def bollinger_value(price, sma, stddev):
        # bb_value[t] = (price[t] - SMA[t])/(2 * stdev[t])
        return (price - sma) / (2 * stddev)

    @staticmethod
    def momentum(price, n):
        # momentum[t] = (price[t]/price[t-N]) - 1
        return price / price.shift(n) - 1

    @staticmethod
    def volatility(price, n):
        # Volatility is just the stdev of daily returns.

        def compute_daily_returns(s):
            # https://www.udacity.com/course/viewer#!/c-ud501/l-4156938722/e-4185858982/m-4185858984
            daily_returns = s.copy()
            daily_returns[1:] = (s[1:] * 1.0 / s[:-1].values) - 1
            return daily_returns[1:]

        return pd.rolling_std(compute_daily_returns(price), n)

    @staticmethod
    def get_bins(data, steps):
        data = data.copy()
        steps -= 1

        stepsize = len(data) / steps
        data.sort()
        threshold = [0] * steps
        for i in range(steps):
            threshold[i] = data[(i + 1) * stepsize]
        return threshold

    @staticmethod
    def to_technical_features(prices, symbol):
        # Bollinger Bands, Momentum, Volatility
        time_window = 20

        sma = pd.rolling_mean(prices, time_window)
        stddev = pd.rolling_std(prices, time_window)

        data = np.ndarray([len(prices), 3])

        data[:, 0] = StrategyLearner.bollinger_value(price=prices, sma=sma, stddev=stddev)[symbol]
        data[:, 1] = StrategyLearner.momentum(price=prices, n=5)[symbol]
        data[1:, 2] = StrategyLearner.volatility(price=prices, n=time_window)[symbol]
        data[:] = np.nan_to_num(data)

        return data

    def build_state(self, array_technical_features, holding):
        state = "".join(map(lambda discrete_value: str(int(discrete_value)), array_technical_features))
        return int(state + str(holding))

    # this method should create a QLearner, and train it for trading
    def addEvidence(self,
                    symbol="IBM",
                    sd=dt.datetime(2008, 1, 1),
                    ed=dt.datetime(2009, 1, 1),
                    sv=10000):

        num_states = 10**4  # 3 features with 9 values, times 3 because buy/sell/hold

        # https://piazza.com/class/ij9yiif53l27fs?cid=1640
        self.learner = ql.QLearner(num_states=num_states,
                                   num_actions=3,  # 3 actions, buy, sell or hold
                                   alpha=0.2,
                                   gamma=0.9,
                                   rar=0.98,
                                   radr=0.999,
                                   dyna=0,
                                   verbose=False)

        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        df = ut.get_data(syms, dates)  # automatically adds SPY
        prices = df[syms]  # only portfolio symbols

        # example use with new colname
        # volume_all = ut.get_data(syms, dates, colname="Volume")  # automatically adds SPY
        # volume = volume_all[syms]  # only portfolio symbols
        # volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        # if self.verbose: print volume

        data = self.to_technical_features(prices, symbol)

        # TODO - Can't I do in one line?
        # TODO - I'd need to store the bins right?
        for i in range(data.shape[1]):
            data[:, i] = np.digitize(x=data[:, i], bins=self.get_bins(data[:, i], 10))

        round_lot = 100
        max_iterations = 100

        cumulative_return = np.ndarray(max_iterations)

        for iteration in range(0, max_iterations):
            # holding - 0
            # buy - 1
            # sell - 2
            df['cash'] = 0
            df['portfolio_value'] = 0
            df['cash'].ix[0] = sv
            df['portfolio_value'].ix[0] = sv

            initial_state = self.build_state(data[0], 0)
            action = self.learner.querysetstate(initial_state)
            prev_date = df.index[0]

            i = 1
            position = 0
            for date in df.index[1:]:
                entry = data[i]
                i += 1

                today_price = prices.ix[date, symbol]

                invalid = False

                if action == 1:  # buy
                    position += 1
                    if position > 1:
                        position = 1
                        invalid = True
                elif action == 2:  # sell
                    position -= 1
                    if position < -1:
                        position = -1
                        invalid = True

                if invalid:
                    action = 0  # TODO: Can I use something like an enum instead of 0, 1 and 2?

                if action == 0:  # hold
                    df.ix[date, 'cash'] = df.ix[prev_date, 'cash']
                elif action == 1:  # buy
                    df.ix[date, 'cash'] = df.ix[prev_date, 'cash'] - today_price * round_lot