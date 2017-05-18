import numpy as np
import pandas as pd
import util
import datetime as dt
import csv

verbose = False
'''
# a simulator for wrapping the learner into a time frame of data features
'''
class Simulator(object):

    def __init__(self, symbols,
        start_date=dt.datetime(2008,1,1),                           #CHANGE DATES TO ACTUAL RANGE!!!
        end_date= dt.datetime(2009,1,1)):

        # frame a time period as world
        self.dates_range = pd.date_range(start_date, end_date)

        # initialize cash holdings
        init_cash = 100000

        #for visualization
        self.data_out = []

        # preprocessing time series
        # stock symbol data
        stock_symbols = symbols[:]
        symbols.append('interest_rates')
        symbols.append('vix')
        # price data
        prices_all = util.get_data(symbols, self.dates_range, True)

        self.stock_A = stock_symbols[0]
        self.stock_B = stock_symbols[1]

        """
        #unemployment rate
        temp_unemp = {}
        unemployment = {}
        with open('unemployment.csv') as unemp_file:
            for line in csv.reader(unemp_file, delimiter=','):
                curr_date = dt.strptime(line[0], '%B-%y')
                temp_unemp[curr_date] = line[1]
        for d in prices_all.keys():
            temp_date = dt.datetime(d.year, d.month)
            if temp_date in temp_unemp:
                unemployment[d] = temp_unemp[temp_date]
        """

        # first trading day
        self.dateIdx = 0
        self.date = prices_all.index[0]
        self.start_date = start_date
        self.end_date = end_date

        self.prices = prices_all[stock_symbols]
        self.prices_SPY = prices_all['spy']
        self.prices_VIX = prices_all['vix']
        self.prices_interest_rate = prices_all['interest_rates']

        # keep track of portfolio value as a series
        self.portfolio = {'cash': init_cash, 'a_vol': [], 'a_price': [], 'b_vol': [], 'b_price': [], 'longA': 0}
        self.port_val = self.port_value_for_output()

        # hardcode enumerating of features
        """
        self.sma = SMA(self.dates_range)
        self.bbp = BBP(self.dates_range)
        self.rsi = RSI(self.dates_range)
        """

    def init_state(self, lookback=50):
        """
        return init states of the market
        """
        states = []
        for _ in range(lookback):
            states.append(self.get_state(self.date))
            self.dateIdx += 1
            self.date = self.prices.index[self.dateIdx]

        return states

    def step(self, action):
        """
        Code indirectly based on previous code written by group member Yiding
        Zhao for the Machine Learning for Trading course
        """
        """
        Take an action, and move the date forward,
        record the reward of the action and date
        action: buy, sell, hold
        return (reward, market status)
        """
        # change state accordingly

        buy_volume = 100
        abs_return_A = 0
        pct_return_A = 0
        abs_return_B = 0
        pct_return_B = 0

        if (action == 'buy'):
            if (self.portfolio['longA'] >= 0):

                if verbose: print('---BUY WITH longA greater/equal 0')
                long_cost = buy_volume * self.prices.ix[self.date, self.stock_A]

                if verbose: print('Buying ' + str(buy_volume) + ' shares of ' + self.stock_A + ' at a price of $' + str(self.prices.ix[self.date, self.stock_A]) + ' per share, for a total cost of $' + str(long_cost) + '.')
                short_cost = buy_volume * self.prices.ix[self.date, self.stock_B]

                if verbose: print('Shorting ' + str(buy_volume) + ' shares of ' + self.stock_B + ' at a price of $' + str(self.prices.ix[self.date, self.stock_B]) + ' per share, for a total cost of $' + str(short_cost) + '.')
                total_cost = short_cost + long_cost

                if verbose: print('Total cost is $' + str(total_cost))

                if verbose: print('Pre-transaction cash is $' + str(self.portfolio['cash']))
                self.portfolio['cash'] -= total_cost

                if verbose: print('Post-transaction cash is $' + str(self.portfolio['cash']))

                self.portfolio['a_vol'].append(buy_volume)
                self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['b_vol'].append(buy_volume)
                self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['longA'] = 1

                if verbose: print(self.portfolio)
                old_port_val = self.port_val
                self.port_val = self.port_value_for_output()

                if verbose: print(self.portfolio)
                reward = self.port_val - old_port_val
                if verbose: print('---END OF BUY WITH longA greater/equal 0')

            else: #longA < 0 --> sell in reverse
                if verbose: print('---BUYING (ACTUALLY SELLING) WITH longA < 0')
                if verbose: print('Selling our long investment of ' + str(self.portfolio['b_vol'][0]) + ' shares for $' + str(self.prices.ix[self.date, self.stock_B]))
                long_initial = self.portfolio['b_vol'][0] * self.portfolio['b_price'][0]
                long_return = self.portfolio['b_vol'].pop(0) * self.prices.ix[self.date, self.stock_B]
                abs_return_B = long_return - long_initial
                pct_return_B = float(abs_return_B) / long_initial
                if verbose:print('Long initial is ' + str(long_initial))
                if verbose:print('Long return is ' + str(long_return))
                if verbose:print('Long return - long initial = ' + str(abs_return_B))
                if verbose:print('(long return - long initial) / long initial = ' + str(pct_return_B))
                self.portfolio['b_price'].pop(0)
                if verbose: print('Return is $' + str(long_return))
                if verbose: print('Cover our long investment of ' + str(self.portfolio['a_vol'][0]) + ' shares that we bought for $' + str(self.portfolio['a_price'][0]) + ' and add to it a gain of $' + str((self.portfolio['a_price'][0] - self.prices.ix[self.date, self.stock_A])) + ' (' + str((self.portfolio['a_price'][0])) + ' - ' + str(self.prices.ix[self.date, self.stock_A]) + ') for each of our ' + str(self.portfolio['a_vol'][0]) + ' stocks.')
                short_initial = self.portfolio['a_vol'][0] * self.portfolio['a_price'][0]
                abs_return_A = (self.portfolio['a_vol'][0] * (self.portfolio['a_price'][0] - self.prices.ix[self.date, self.stock_A]))
                short_return = self.portfolio['a_vol'][0] * self.portfolio['a_price'][0]
                short_return += (self.portfolio['a_vol'].pop(0) * (self.portfolio['a_price'].pop(0) - self.prices.ix[self.date, self.stock_A]))
                pct_return_A = float(abs_return_A) / short_initial
                if verbose:print('Short initial is ' + str(short_initial))
                if verbose:print('Short return is ' + str(short_return))
                if verbose:print('Absolute return for short is ' + str(abs_return_A))
                if verbose:print('Percetn return for short is ' + str(pct_return_A))
                if verbose: print('Short return is $' + str(short_return))
                if verbose: print('Old cash is $' + str(self.portfolio['cash']))
                new_cash = self.portfolio['cash'] + long_return + short_return
                self.portfolio['cash'] = new_cash
                if verbose: print('New cash is $' + str(self.portfolio['cash']))
                self.portfolio['longA'] = -1 if (len(self.portfolio['a_vol']) > 0) else 0
                old_port_val = self.port_val
                self.port_val = self.port_value_for_output()
                if verbose: print(self.portfolio)
                reward = self.port_val - old_port_val
                if verbose: print('Old portfolio value is $' + str(old_port_val))
                if verbose: print('New portfolio value is $' + str(self.port_val))
                if verbose: print('Reward is $' + str(reward))

        elif (action == 'sell'):
            if (self.portfolio['longA'] > 0):
                if verbose: print('---SELLING WITH longA > 0')
                if verbose: print('Selling our long investment of ' + str(self.portfolio['a_vol'][0]) + ' shares for $' + str(self.prices.ix[self.date, self.stock_A]))
                long_initial = self.portfolio['a_vol'][0] * self.portfolio['a_price'][0]
                long_return = self.portfolio['a_vol'].pop(0) * self.prices.ix[self.date, self.stock_A]
                abs_return_A = long_return - long_initial
                pct_return_A = float(abs_return_A) / long_initial
                self.portfolio['a_price'].pop(0)
                if verbose: print('Return is $' + str(long_return))
                if verbose: print('Cover our long investment of ' + str(self.portfolio['b_vol'][0]) + ' shares that we bought for $' + str(self.portfolio['b_price'][0]) + ' and add to it a gain of $' + str((self.portfolio['b_price'][0] - self.prices.ix[self.date, self.stock_B])) + ' (' + str((self.portfolio['b_price'][0])) + ' - ' + str(self.prices.ix[self.date, self.stock_B]) + ') for each of our ' + str(self.portfolio['b_vol'][0]) + ' stocks.')
                short_initial = self.portfolio['b_vol'][0] * self.portfolio['b_price'][0]
                abs_return_B = (self.portfolio['b_vol'][0] * (self.portfolio['b_price'][0] - self.prices.ix[self.date, self.stock_B]))
                short_return = self.portfolio['b_vol'][0] * self.portfolio['b_price'][0]
                short_return += (self.portfolio['b_vol'].pop(0) * (self.portfolio['b_price'].pop(0) - self.prices.ix[self.date, self.stock_B]))
                pct_return_B = float(abs_return_B) / short_initial
                if verbose: print('Short return is $' + str(short_return))
                if verbose: print('Old cash is $' + str(self.portfolio['cash']))
                new_cash = self.portfolio['cash'] + long_return + short_return
                self.portfolio['cash'] = new_cash
                if verbose: print('New cash is $' + str(self.portfolio['cash']))
                self.portfolio['longA'] = 1 if (len(self.portfolio['a_vol']) > 0) else 0
                old_port_val = self.port_val
                self.port_val = self.port_value_for_output()
                if verbose: print(self.portfolio)
                reward = self.port_val - old_port_val
                if verbose: print('Old portfolio value is $' + str(old_port_val))
                if verbose: print('New portfolio value is $' + str(self.port_val))
                if verbose: print('Reward is $' + str(reward))
            else: # longA <= 0 --> buy in reverse
                if verbose: print('---SELLING (ACTUALLY BUYING) WITH long <= 0')
                long_cost = buy_volume * self.prices.ix[self.date, self.stock_B]
                if verbose: print('Buying ' + str(buy_volume) + ' shares of ' + self.stock_B + ' at a price of $' + str(self.prices.ix[self.date, self.stock_B]) + ' per share, for a total cost of $' + str(long_cost) + '.')
                short_cost = buy_volume * self.prices.ix[self.date, self.stock_A]
                if verbose: print('Shorting ' + str(buy_volume) + ' shares of ' + self.stock_A + ' at a price of $' + str(self.prices.ix[self.date, self.stock_A]) + ' per share, for a total cost of $' + str(short_cost) + '.')
                total_cost = short_cost + long_cost
                if verbose: print('Total cost is $' + str(total_cost))
                if verbose: print('Pre-transaction cash is $' + str(self.portfolio['cash']))
                self.portfolio['cash'] -= total_cost
                if verbose: print('Post-transaction cash is $' + str(self.portfolio['cash']))
                self.portfolio['a_vol'].append(buy_volume)
                self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['b_vol'].append(buy_volume)
                self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['longA'] = -1
                if verbose: print(self.portfolio)
                old_port_val = self.port_val
                self.port_val = self.port_value_for_output()
                if verbose: print(self.portfolio)
                reward = self.port_val - old_port_val
                if verbose: print('---END OF SELLING (ACTUALLY BUYING) WITH long <= 0')
        else: #hold
            if verbose: print('---HOLDING')
            old_port_val = self.port_val
            self.port_val = self.port_value_for_output()
            if verbose: print(self.portfolio)
            reward = self.port_val - old_port_val
        #self.port_val = self.port_value_for_output()
        print "port value", self.port_val
        self.data_out.append(self.date.isoformat()[0:10] + ',' + str(self.prices.ix[self.date, self.stock_A]) + ',' + str(self.prices.ix[self.date, self.stock_B]) + ',' + action + ',' + str(abs_return_A) + ',' +  str(pct_return_A) + ',' + str(abs_return_B) + ',' + str(pct_return_B) + ',' + str((self.prices.ix[self.date, self.stock_A] - self.prices.ix[self.date, self.stock_B])) + ',' + str(self.prices_interest_rate[self.date]) + ',' + str(self.prices_SPY[self.date]) + ',' + str(self.prices_VIX[self.date]) + ',' + str(self.port_val))
        if verbose: print(self.data_out)
        state = self.get_state(self.date)
        if verbose: print(state)
        self.dateIdx += 1
        if self.dateIdx < len(self.prices.index):
            self.date = self.prices.index[self.dateIdx]
        if verbose: print(self.get_state(self.date))
        if verbose: print('New date is')
        if verbose: print(self.date)
        if verbose: print('Reward is ' + str(reward))
        return (reward, state)

    def get_state(self, date):
        """
        return state of the market, i.e. prices of certain symbols,
        number of shares hold
        """
        if date not in self.dates_range:
            if verbose: print('Date was out of bounds.')
            if verbose: print(date)
            exit

        # a vector of features
        if (date == self.prices.index[-1]):
            file_name = "data_for_vis_%s.csv" % dt.datetime.now().strftime("%H-%M-%S")
            print "saving to", file_name
            file = open(file_name, 'w');
            for line in self.data_out:
                file.write(line);
                file.write('\n')
            file.close()
        return [self.prices.ix[date, self.stock_A]/self.prices.ix[0, self.stock_A] - self.prices.ix[date, self.stock_B]/self.prices.ix[0, self.stock_B],
            self.prices_interest_rate[date]/self.prices_interest_rate[0] - 1,
            self.prices_SPY[date]/self.prices_SPY[0] - 1,
            self.prices_VIX[date]/self.prices_VIX[0] - 1,
            self.port_val / 100000.0 - 1,
            ]

    # calculate the current value of cash and stock holdings
    def port_value(self):
        value = self.portfolio['cash']
        if (len(self.portfolio['a_vol']) > 0):
            for i in range(len(self.portfolio['a_vol'])):
                value += (self.portfolio['a_vol'][i] * self.portfolio['a_price'][i])
        if (len(self.portfolio['b_vol']) > 0):
            for i in range(len(self.portfolio['b_vol'])):
                value += (self.portfolio['b_vol'][i] * self.portfolio['b_price'][i])
        return value

    # alternate calculation of the current value of cash and stock holdings
    def port_value_for_output(self):
        value = self.portfolio['cash']
        if (self.portfolio['longA'] > 0):
            value += (sum(self.portfolio['a_vol']) * self.prices.ix[self.date, self.stock_A])
            for i in range(len(self.portfolio['b_vol'])):
                value += (self.portfolio['b_vol'][i] * self.portfolio['b_price'][i])
                value += (self.portfolio['b_vol'][i] * (self.portfolio['b_price'][i] - self.prices.ix[self.date, self.stock_B]))
        if (self.portfolio['longA'] < 0):
            value += (sum(self.portfolio['b_vol']) * self.prices.ix[self.date, self.stock_B])
            for i in range(len(self.portfolio['a_vol'])):
                value += (self.portfolio['a_vol'][i] * self.portfolio['a_price'][i])
                value += (self.portfolio['a_vol'][i] * (self.portfolio['a_price'][i] - self.prices.ix[self.date, self.stock_A]))
        return value

    def has_more(self):
        if ((self.dateIdx < len(self.prices.index)) == False):
            print('\n\n\n*****')
            print(self.baseline())
            print('*****\n\n\n')
        return self.dateIdx < len(self.prices.index)

    def baseline(self):
        num_shares = 100000 / self.prices_SPY[0]
        return num_shares * self.prices_SPY[-1]

def main():
    # TO DO
    sim = Simulator('Hi, Yiding')
    sim.init_state()
    #sim.get_state(sim.prices.index[-1])
    sim.step('buy')
    sim.step('hold')
    sim.step('buy')
    sim.step('sell')
    sim.step('sell')
    sim.step('hold')
    sim.step('hold')
    sim.step('sell')
    sim.step('sell')
    sim.step('buy')
    sim.step('buy')


if __name__ == '__main__':
    main()
