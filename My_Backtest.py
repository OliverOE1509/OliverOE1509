import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl, plt 
from datetime import datetime
from dateutil import relativedelta
import yfinance as yf

#plt.style.use('seaborn') 
#mpl.rcParams['font.family'] = 'serif'

pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 200)


class BacktestBase(object):
	def __init__(self, symbol, start, end, amount, ftc=0.0, ptc=0.0, verbose = False):
		self.symbol = symbol
		self.start = start
		self.end = end
		self.initial_amount = amount
		self.amount = amount
		self.ftc = ftc
		self.ptc = ptc
		self.units = 0
		self.position = 0
		self.trades = 0
		self.verbose = verbose
		self.equity_curve = []
		self.download_data()
		self.tp = 0
		self.sl = 0
		#self.winning_trades = []
	

	def create_connection(self):
		conn = None
		try:
			conn = sqlite3.connect('FinansDB_lokal.db')
		except error as e:
			print(e)
		return conn

	def format_connection(self):
		'''Utfører sql spørring for å hente ut data til symbol'''
		connection = self.create_connection()
		aksje_df = pd.read_sql(f'SELECT * FROM AksjeKurser WHERE symbol = "{str(self.symbol)}"', connection, index_col = 'Date')
		connection.close()
		#aksje_df.drop(['ID'], axis=1, inplace=True)
		aksje_df.rename(columns = {'Open_': 'Open', 'Close_': 'Close', 'Adj_Close': 'Adj Close'}, inplace=True)
		return aksje_df

	def get_data(self):
		'''formaterer data etter bestemt dato intervall, skifter navn av Close til price, og kalkulerer log(avkastning)'''
		raw = self.format_connection()
		raw = raw.loc[self.start:self.end]
		raw['return'] = np.log(raw.Close / raw.Close.shift(1))
		self.data = raw.dropna()


	def download_data(self):
		hist_data = yf.download(str(self.symbol), start_date = self.start, end_date = self.end)
		hist_data['return'] = hist_data['Adj Close'].pct_change()
		self.data = hist_data.dropna()



	def plot_data(self, cols=None):
		'''Plots the closing prices for symbol'''
		if cols is None:
			cols = ['price']
		self.data['Close'].plot(figsize=(10, 6), title = self.symbol)

	def get_date_price(self, bar):
		'''Return date and price for a bar'''
		date = str(self.data.index[bar])[:10]
		price = self.data.Close.iloc[:bar]
		return date, price

	def print_balance(self, bar):
		'''Prints out the current cash balance'''
		date, Close = self.get_date_price(bar)
		print(f'{date} | current balance {self.amount:.2f}')

	def print_net_wealth(self, bar):
		'''print current running net wealth'''
		date, Close = self.get_date_price(bar)
		net_wealth = self.units * Close + self.amount
		#print(f'{date} | current net wealth {round(net_wealth, 2)}')
		return date, net_wealth

	def place_buy_order(self, bar, units=None, amount=None):
		'''Place buy order'''
		sl = None
		tp = None
		date, Close = self.get_date_price(bar)
		#self.winning_trades.append((Close))
		if units is None:
			units = int(amount / Close)
		self.amount -= (units * Close) * (1 + self.ptc ) + self.ftc
		self.units += units
		self.trades += 1
		#print(self.print_balance(bar))
		if self.verbose:
			print(f'{date} |  {units} at price {price:.2f}')
			self.print_balance()
			self.print_net_wealth()

	def place_sell_order(self, bar, units=None, amount=None):
		'''Place sell order'''
		date, Close = self.get_date_price(bar)
		if units is None:
			units = int(amount / Close)
		self.amount += (units * Close) * (1 - self.pct) - self.ftc
		self.equity_curve.append(self.amount)
		#print(self.equity_curve)
		self.units -= units
		self.trades += 1
		if self.verbose:
			print(f'{date} | selling {units} at price {price:.2f}')
			self.print_balance()
			self.print_net_wealth()

	def close_out(self, bar):
		'''closing out position at end of backtesting period'''
		date, Close = self.get_date_price(bar)
		self.amount += self.units * Close
		self.units = 0
		self.trades += 1
		buy_hold = ((self.data.Close[-1] - self.data.Close[0]) / self.data.Close[0] * 100)

		delta = relativedelta.relativedelta(datetime.strptime(self.end, '%Y-%m-%d'), datetime.strptime(self.start, '%Y-%m-%d')).years


		cagr = ((self.amount / self.initial_amount)**(1/delta) - 1) * 100
		if self.verbose:
			print(f'{date} | inventory {self.units} units at {Close:.2f}')
			print('=' * 55)
		print('Start balance [NOK] {:.2f}'.format(self.initial_amount))
		print('Final balance  [NOK] {:.2f}'.format(self.amount))
		perf = ((self.amount - self.initial_amount) /
			self.initial_amount * 100)
		print('Net performance [%] {:.2f}'.format(perf))
		print('Buy & Hold performance [%] {:.2f}'.format(buy_hold))
		print('CAGR [%] {:.2f}'.format(cagr))
		print('Trades executed [#] {:.2f}'.format(self.trades))
		print('=' * 55)
		#print(self.winning_trades)
		#self.equity_curve.append(self.amount)


if __name__ == '__main__':
	bb = BacktestBase('EQNR.OL', '2020-01-01', '2022-01-01', 10000)
	#print(bb.data.info())
	bb.plot_data()
	#bb.plot_amount()

	plt.show()




