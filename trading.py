from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt


class TestStrategy(bt.Strategy):
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None

        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=15)
        self.rsi = bt.indicators.RelativeStrengthIndex()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        self.log('Close, %.2f' % self.dataclose[0])
        print('rsi:', self.rsi[0])
        if self.order:
            return

        if not self.position:
            if (self.rsi[0] < 30):
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.order = self.buy(size=500)

        else:
            if (self.rsi[0] > 70):
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                self.order = self.sell(size=500)


class TestBuySell(bt.Strategy):
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.data_close1 = self.datas[0].close
        self.data_close2 = self.datas[1].close
        self.order = None
        self.buyprice = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
                #
                # self.buyprice = order.executed.price
                # self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        self.log('Close, %.2f' % self.data_close1[0])
        self.log('Close, %.2f' % self.data_close2[0])
        if self.order:
            return

        if self.data_close2[0] > self.data_close1[0]:
            self.log('BUY CREATE, %.2f' % self.data_close1[0])
            self.order = self.buy(size=500)
        elif self.data_close2[0] < self.data_close1[0]:
            self.log('SELL CREATE, %.2f' % self.data_close1[0])
            self.order = self.sell(size=500)


if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TestBuySell)

    # Analyzer
    # cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='mysharpe')

    #cerebro.broker.setcommission(commission=0.001)

    datapath1 = 'Data/AAPL_2018_2019.csv'
    datapath2 = 'Data/DJIA_2018_2019.csv'

    # Create a Data Feed
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath1,
        fromdate=datetime.datetime(2018, 1, 1),
        todate=datetime.datetime(2019, 1, 1),
        reverse=False)

    data2 = bt.feeds.YahooFinanceCSVData(
        dataname=datapath2,
        fromdate=datetime.datetime(2018, 1, 1),
        todate=datetime.datetime(2019, 1, 1),
        reverse=False)

    cerebro.adddata(data)
    cerebro.adddata(data2)
    cerebro.broker.setcash(100000.0)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.plot()

