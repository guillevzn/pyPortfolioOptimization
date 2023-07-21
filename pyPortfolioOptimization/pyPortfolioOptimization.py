import numpy as np
from pandas_datareader import data as pdr
import datetime as dt
import yfinance as yf

class pyPortfolioOptimization:
    called_getData = False
    def __init__(self, stocks, start, end):
        yf.pdr_override()
        self.stocks = stocks
        self.start = start
        self.end = end

    def getData(self):
        stockData = pdr.get_data_yahoo(self.stocks, start=self.start, end=self.end)
        stockData = stockData['Close']

        returns = stockData.pct_change()
        meanReturns = returns.mean()
        covMatrix = returns.cov()

        self.meanReturns = meanReturns
        self.covMatrix = covMatrix

        pyPortfolioOptimization.called_getData = True

        return meanReturns, covMatrix

    def portfolioPerformance(self, weights):
        if pyPortfolioOptimization.called_getData == True:
            pass
        else:
            meanReturns, covMatrix = self.getData()
            self.meanReturns = meanReturns
            self.covMatrix = covMatrix
        
        returns = np.sum(self.meanReturns*weights)*252
        std = np.sqrt(
                np.dot(weights.T,np.dot(self.covMatrix, weights))
            )*np.sqrt(252)
        return returns, std
