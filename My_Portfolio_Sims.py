import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from functools import reduce

from pypfopt import HRPOpt
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage, risk_matrix
from pypfopt.efficient_frontier import EfficientFrontier,EfficientCVaR
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

from scipy.stats import shapiro 
from scipy.stats import kstest


pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1000)
#pd.set_option('display.max_rows', 10000)

class My_Portfolio:
	def __init__(self, symbol, train_start, train_end, test_start, test_end, initial_capital):
		self.symbol = symbol

		if train_start:
			self.train_start = train_start
		if train_end:
			self.train_end = train_end
		if test_start:
			self.test_start = test_start
		if test_end:
			self.test_end = test_end

		self.capital = initial_capital

		self.fetch_data()
		self.training_data()
		self.testing_data()

	
	def fetch_data(self):
		data_frames = []
		for sym in self.symbol:
			data = yf.download(sym, start = self.train_start, end=self.test_end)
			data[f'{sym}'] = data["Adj Close"]
			data = data[[f'{sym}']]
			data_frames.append(data)

		self.data = reduce(lambda  left,right: pd.merge(left,right,on=['Date'], how='outer'), data_frames).dropna()

	def training_data(self):
		self.train_data = self.data.loc[self.train_start:self.train_end]

	def testing_data(self):
		self.test_data = self.data.loc[self.test_start:self.test_end]

	def simulate_Portfolio(self, tickers_dict):
		'''Makes a Dataframe with selected tickers as a Dictioanary where key 
		is ticker and value is number of stocks Dates used in this function is 
		from 1.6.22 until today Returns a dataframe with selected stocks'''

		to_be_invested = self.test_data[tickers_dict.keys()]

		investment = to_be_invested.mul(pd.Series(tickers_dict),axis=1)
		return investment

	def Mvo(self):
		mu = mean_historical_return(self.train_data)
		S = CovarianceShrinkage(self.train_data).ledoit_wolf()
		ef = EfficientFrontier(mu, S) # Instantiated EfficientFrontier Object
		weights = ef.max_sharpe() # Must be performed to calculate clean_weights
		cleaned_weights = ef.clean_weights()
		#sharpe_CVar = ef.portfolio_performance(verbose=True)[2]
		latest_prices = get_latest_prices(self.train_data)
		da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=self.capital)

		allocation1, leftover1 = da.greedy_portfolio()
		#print("Discrete allocation:", allocation1)
		#print("Funds remaining: NOK {:.2f}".format(leftover1))
		port1 = self.simulate_Portfolio(allocation1)
		port1 = port1.sum(axis=1)
		ROI_port1 = ((port1.iloc[-1] - port1.iloc[0]) / port1.iloc[0] ) * 100
		return port1



	def Hrp(self):
		returns = self.train_data.pct_change().dropna()
		hrp = HRPOpt(returns)
		hrp_weights = hrp.optimize()
		#summary_hrp = hrp.portfolio_performance(verbose=True)
		#sharpe_hrp = summary_hrp[2]
		latest_prices = get_latest_prices(self.train_data)
		da_hrp = DiscreteAllocation(hrp_weights, latest_prices, total_portfolio_value=self.capital)

		allocation2, leftover2 = da_hrp.greedy_portfolio()
		port2 = self.simulate_Portfolio(allocation2)
		port2 = port2.sum(axis=1)
		ROI_port2 = ((port2.iloc[-1] - port2.iloc[0]) / port2.iloc[0] ) * 100
		return port2

	def mCVAR(self):
		mu = mean_historical_return(self.train_data)
		S = CovarianceShrinkage(self.train_data).ledoit_wolf()
		ef_cvar = EfficientCVaR(mu, S)
		cvar_weights = ef_cvar.min_cvar()

		cleaned_weights = ef_cvar.clean_weights()
		latest_prices = get_latest_prices(self.train_data)
		da_cvar = DiscreteAllocation(cvar_weights, latest_prices, total_portfolio_value=self.capital)

		allocation3, leftover3 = da_cvar.greedy_portfolio()

		port3 = self.simulate_Portfolio(allocation3)
		port3 = port3.sum(axis=1)
		return port3

	def plot_trained_portfolio(self):
		# Will compare the result with the three methods:
		p1 = self.Mvo()
		p2 = self.Hrp()
		p3 = self.mCVAR()

		frame = {'MVO':p1, 'HRP':p2, 'mCVAR':p3}
		compare_df = pd.DataFrame(data = frame)
		compare_df.plot(legend=True, figsize=(10,5))
		plt.show()



# Example dates:
train_start_date = datetime.datetime(2010,1,1)
train_end_date = datetime.datetime(2017, 12, 31)

test_start_date = datetime.datetime(2018,1,1)
test_end_date = datetime.datetime.today()

tickers = ['VGLT', 'VTI', 'BIV', 'GLD', 'USCI']


portfolio = My_Portfolio(symbol = tickers, 
	train_start = train_start_date, 
	train_end = train_end_date, 
	test_start = test_start_date, 
	test_end = test_end_date,
	initial_capital = 100000)


data = portfolio.plot_trained_portfolio()


