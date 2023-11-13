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
		self.buy_trades = {'Date': [], 'buy_Close': []}
		self.sell_trades = {'Date': [], 'sell_Close': []}
		print("Er i backtestBase objekt")


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
		hist_data = yf.download(str(self.symbol), start = self.start, end = self.end)
		hist_data['return'] = hist_data['Adj Close'].pct_change()
		self.data = hist_data.dropna()
		#print(type(self.data.index))



	def plot_data(self, cols=None):
		'''Plots the closing prices for symbol'''
		eq = self.get_equity_curve()

		labels = self.get_entry_exit_pts()
		if len(labels.index) > 1:
			buy_labels = labels[labels['buy_Close'] != 0]['buy_Close']
			sell_labels = labels[labels['sell_Close'] != 0]['sell_Close']

			fig, (ax1, ax2) = plt.subplots(2, figsize = (12,10))
			if cols is None:
				cols = ['price']
			ax1.plot(self.data.index, self.data['Close'], label = "Close Price", color = 'black', alpha = 0.7)
			ax1.scatter(buy_labels.index, buy_labels, color = 'green', label = 'buy', marker = '^')
			ax1.scatter(sell_labels.index, sell_labels, color = 'red', label = 'sell', marker = 'v', alpha = 0.5)
			#ax1.ylabel(f'Price {self.symbol}')
			#ax1.title("Close price with buy and sell labels")

			ax2.plot(eq.index, eq.loc[:, 'net_wealth'], color = 'black', label = 'Equity Curve')
		else:
			print("There are no labels -> No signals detected")


	def get_date_price(self, bar):
		'''Return date and price for a bar'''
		date = str(self.data.index[bar])[:10]
		price = self.data.Close.iloc[bar]
		return date, price

	def print_balance(self, bar):
		'''Prints out the current cash balance'''
		date, Close = self.get_date_price(bar)
		print(f'{date} | current balance {self.amount}')
		#return {'date' : date, 'balance': self.amount}

	def print_net_wealth_buy(self, bar):
		'''print current running net wealth'''
		date, Close = self.get_date_price(bar)
		net_wealth = self.units * Close + self.amount
		leftover = int((self.units * Close) / Close)
		print(f'{date} | current net wealth {net_wealth}')
		return {'date': date, 'net_wealth': net_wealth}

	def get_net_wealth_buy(self, bar):
		'''print current running net wealth'''
		date, Close = self.get_date_price(bar)
		net_wealth = self.units * Close + self.amount
		leftover = int((self.units * Close) / Close)
		return {'date': date, 'net_wealth': net_wealth}

	def place_buy_order(self, bar, units=None, amount=None):
		'''Place buy order'''
		date, Close = self.get_date_price(bar)
		self.buy_trades['Date'].append(date)
		self.buy_trades['buy_Close'].append(Close)

		if units is None:
			units = int(amount / Close)

		self.amount -= (units * Close) * (1 + self.ptc ) + self.ftc
		self.units += units
		self.trades += 1
		self.position = 1
		self.equity_curve.append(( self.get_net_wealth_buy(bar)))
		#self.equity_curve.append({'date': date, 'net_wealth': self.amount})
		if self.verbose:
			print("-"*50)
			print(f'Beginning trade nr: {len(self.buy_trades["buy_Close"])}')
			print(f'{date} | Buying {self.units} at price {Close:.2f}')
			#self.print_balance(bar)
			self.print_net_wealth_buy(bar)
			'''We only use print_net_wealth_buy in buy orders because we only get the remaining balance after our discrete amount of securities have been bought at price x'''
			print("="*50)
			print()

	def place_sell_order(self, bar, units=None, amount=None):
		'''Place sell order'''
		date, Close = self.get_date_price(bar)
		self.sell_trades['Date'].append(date)
		self.sell_trades['sell_Close'].append(Close)

		if units is None:
			units = int(amount / Close)

		self.amount += (units * Close) * (1 - self.ptc) - self.ftc
		self.trades += 1
		self.position = 0
		#self.equity_curve.append(( self.print_balance(bar)))
		self.equity_curve.append({'date': date, 'net_wealth': self.amount})

		if self.verbose:
			print("-"*50)
			print(f'Ending trade nr: {len(self.buy_trades["buy_Close"])}')
			print(f'{date} | Selling {self.units} at price {Close:.2f}')
			self.print_balance(bar)
			'''Vi bruker kun print_balance i salgsordrer, fordi vi ikke ønsker å legge til det opprinnelige beløpet etter at vi har solgt.'''
			#self.print_net_wealth_buy(bar)
			print("="*50)
			print("*"*50)
		self.units -= units


	def get_entry_exit_pts(self):
		try: 
			buy_trades_df = pd.DataFrame.from_dict(self.buy_trades).set_index("Date")
			sell_trades_df = pd.DataFrame.from_dict(self.sell_trades).set_index("Date")
			
		except Error as e:
			print(e)

		else:
			trades_df = pd.concat([buy_trades_df, sell_trades_df], ignore_index = False, axis=0).sort_values(by="Date").fillna(0)
			trades_df.index = pd.to_datetime(trades_df.index)
			return trades_df
			
		

	def get_win_rate(self):	
		trades_df = self.get_entry_exit_pts()
		frames = trades_df['buy_Close'] + trades_df['sell_Close']

		every_scnd_from_zero = frames[0::2]
		every_scnd_from_one = frames[1::2]
		pairwise_trades = list(zip(every_scnd_from_one, every_scnd_from_zero))
		win_count = 0
		for i in pairwise_trades:
			diff = i[0] - i[1]
			if diff > 0:
				win_count += 1
		return win_count

	def get_equity_curve(self):
		try:
			eq_curv = pd.DataFrame.from_dict(self.equity_curve).set_index('date')
		except:
			print("No equity curve")
		else:
			eq_curv.index = pd.to_datetime(eq_curv.index)
			return eq_curv



	def close_out(self, bar):
		'''closing out position at end of backtesting period'''
		date, Close = self.get_date_price(bar)
		if self.position == 1:
			self.sell_trades['Date'].append(date)
			self.sell_trades['sell_Close'].append(Close)

		self.amount += self.units * Close
		self.units = 0
		self.trades += 1
		if self.verbose:
			print(f'{date} | Selling {self.units} at price {Close:.2f}')
			print("-"*50)
			self.print_balance(bar)
			self.print_net_wealth_buy(bar)
			print("="*50)
			print("*"*50)


		buy_hold = ((self.data.Close[-1] - self.data.Close[0]) / self.data.Close[0] * 100)

		delta = relativedelta.relativedelta(datetime.strptime(self.end, '%Y-%m-%d'), datetime.strptime(self.start, '%Y-%m-%d')).years


		cagr = ((self.amount / self.initial_amount)**(1/delta) - 1) * 100
		#if self.verbose:
			#print(f'{date} | inventory {self.units} units at {Close:.2f}')
			#print('=' * 55)
		print('Start balance [NOK] {:.2f}'.format(self.initial_amount))
		print('Final balance  [NOK] {:.2f}'.format(self.amount))

		performance = ((self.amount - self.initial_amount) /
			self.initial_amount * 100)
		print('Net performance [%] {:.2f}'.format(performance))
		print('Buy & Hold performance [%] {:.2f}'.format(buy_hold))
		print('CAGR [%] {:.2f}'.format(cagr))
		print('Trades executed [#] {:.2f}'.format(self.trades))
		print('Win rate [%] {:.2f}'.format((self.get_win_rate() / len(self.buy_trades['Date']))*100))
		print('=' * 55)
		#print(self.sell_trades)
		#print("Equity Curve", self.equity_curve)
		#print(self.get_equity_curve())

		#self.equity_curve.append(self.amount)


if __name__ == '__main__':
	pass
	#bb = BacktestBase('EQNR.OL','2010-01-01', '2022-01-01', 10000)
	



