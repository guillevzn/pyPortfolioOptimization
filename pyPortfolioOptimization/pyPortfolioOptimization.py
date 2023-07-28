import numpy as np
import pandas as pd
import scipy as sc
import yfinance as yf
import plotly.express as px
import plotly.graph_objs as go

class pyPortfolioOptimization:

    called_getData = False

    def __init__(self, stocks, start, end):
        yf.pdr_override()
        self.stocks = stocks
        self.start = start
        self.end = end


    def _portfolioReturns(self, weights, meanReturns):
        '''
        Portfolio returns
        '''
        return np.sum(meanReturns*weights)*self.len_period


    def _portfolioVar(self, weights, covMatrix):
        '''
        Portfolio variance
        '''
        return np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(self.len_period)
    

    def _efficientOpt(self, meanReturns, covMatrix, returnTarget, constraintSet):
        '''
        For each returnTarget, we want to optimise the portfolio for min variance.
        '''
        numAssets = len(meanReturns)
        args = (covMatrix)

        constraints = ({'type':'eq', 'fun': lambda x: self._portfolioReturns(x, meanReturns) - returnTarget},
                        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = constraintSet
        bounds = tuple(bound for asset in range(numAssets))
        effOpt = sc.optimize.minimize(self._portfolioVar, [1./numAssets]*numAssets, args=args, method = 'SLSQP', bounds=bounds, constraints=constraints)
        return effOpt
    

    # Download Risk Free Rate
    '''
    Download 13 week us treasury bills rates
    and convert to daily rate.
    '''
    def getRiskFreeRate(self):

        def deannualize(annual_rate, periods=365):
            return (1 + annual_rate) ** (1/periods) - 1

        annualized = yf.download('^IRX', start=self.start, end=self.end)['Close']

        daily = annualized.apply(deannualize)
        return daily.mean()


    # Check if getData
    '''
    Check if getData has already been executed or not.
    '''
    def check_getData(self):
        if pyPortfolioOptimization.called_getData == True:
            pass
        else:
            meanReturns, covMatrix = self.getData()
            self.meanReturns = meanReturns
            self.covMatrix = covMatrix


    # Importing data
    '''
    Get data from yfinance and calculate
    mean returns and covariance matrix.
    It stores returned data.
    '''
    def getData(self):
        stockData = yf.download(self.stocks, start=self.start, end=self.end)
        self.stockData = stockData
        self.len_period = len(stockData)
        stockData = stockData['Close']

        returns = stockData.pct_change()
        meanReturns = returns.mean()
        covMatrix = returns.cov()

        self.meanReturns = meanReturns
        self.covMatrix = covMatrix

        pyPortfolioOptimization.called_getData = True

        return meanReturns, covMatrix


    # Plot data
    '''
    Plot downloaded data.
    '''
    def plotData(self):
        self.check_getData()
        
        pd.options.plotting.backend = "plotly"

        fig = self.stockData['Close'].plot()

        fig.update_layout(
            title='Close values for stocks',
            yaxis = dict(title='Daily value'),
            xaxis = dict(title='Date (days)'),
            showlegend = True,
            legend = dict(
                x = 1.05, y = 0, traceorder='normal',
                bgcolor='#E2E2E2',
                bordercolor='black',
                borderwidth=2),
            width=1000,
            height=600
        )

        return fig.show()


    # Maximize for the Sharpe Ratio
    '''
    Highest ratio for minimum volatility.
    '''
    def maxSharpeRatio(self, riskFreeRate = 0, constraintSet=(0,1)):

        '''
        Set 13 week treasury bill rate as free-risk ratio,
        from the same dates as the selected stocks.
        '''
        if riskFreeRate == '13-week':
            riskFreeRate = self.getRiskFreeRate()

        def objFunc(weights, meanReturns, covMatrix, riskFreeRate):
            '''
            Objective function: Negative Sharpe ratio
            '''
            funcDenomr = np.sqrt(np.matmul(np.matmul(weights, covMatrix), weights.T) )
            funcNumer = np.matmul(np.array(meanReturns),weights.T)-riskFreeRate
            return -(funcNumer / funcDenomr)
        
        def constraintEq(weights):
            '''
            Constraint equations
            '''
            A=np.ones(weights.shape)
            b=1
            return np.matmul(A,weights.T)-b 
        
        self.check_getData()
        '''
        We want to minimize the negative Sharpe ratio, instead of maximazing
        the positive Sharpe ratio.
        Minimize the negative SR, by altering the weights of the portfolio.
        '''
        numAssets = len(self.meanReturns)
        args = (self.meanReturns, self.covMatrix, riskFreeRate)
        constraints = ({'type': 'eq', 'fun': constraintEq})
        bound = constraintSet
        bounds = tuple(bound for asset in range(numAssets))
        result = sc.optimize.minimize(objFunc, [1./numAssets]*numAssets, args=args,
                            method='SLSQP', bounds=bounds, constraints=constraints)
        result.x = [float('{:.6f}'.format(val)) for val in result.x]
        result.jac = [float('{:.6f}'.format(val)) for val in result.jac]
        return result
    

    # Minimium for the Portfolio Variance
    '''
    Minimize the portfolio variance by altering the 
    weights/allocation of assets in the portfolio.
    '''
    def minimizeVariance(self, constraintSet=(0,1)):
        
        self.check_getData()

        numAssets = len(self.meanReturns)
        args = (self.covMatrix)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = constraintSet
        bounds = tuple(bound for asset in range(numAssets))
        result = sc.optimize.minimize(self._portfolioVar, numAssets*[1./numAssets], args=args,
                            method='SLSQP', bounds=bounds, constraints=constraints)
        result.x = [float('{:.6f}'.format(val)) for val in result.x]
        result.jac = [float('{:.6f}'.format(val)) for val in result.jac]
        return result


    # Efficient Frontier
    '''
    Calculate the efficient frontier of allocations.
    '''
    def efficientFrontier(self, iterations, riskFreeRate=0, constraintSet=(0,1)):
        # , meanReturns, covMatrix
        '''
        Set 13 week treasury bill rate as free-risk ratio,
        from the same dates as the selected stocks.
        '''
        if riskFreeRate == '13-week':
            riskFreeRate = self.getRiskFreeRate()

        self.check_getData()

        '''
        Read in mean, cov matrix, and other financial information.
        Output, Max SR , Min Volatility, efficient frontier.
        '''
        # Max Sharpe Ratio Portfolio
        maxSR_Portfolio = np.array(self.maxSharpeRatio()['x'])
        maxSR_returns, maxSR_std = self._portfolioReturns(maxSR_Portfolio, self.meanReturns), self._portfolioVar(maxSR_Portfolio, self.covMatrix)
        maxSR_allocation = pd.DataFrame(maxSR_Portfolio, index=self.meanReturns.index, columns=['allocation'])
        maxSR_allocation.allocation = [round(i*100,0) for i in maxSR_allocation.allocation]
        
        # Min Volatility Portfolio
        minVol_Portfolio = np. array(self.minimizeVariance()['x'])
        minVol_returns, minVol_std = self._portfolioReturns(minVol_Portfolio, self.meanReturns), self._portfolioVar(minVol_Portfolio, self.covMatrix)
        minVol_allocation = pd.DataFrame(minVol_Portfolio, index=self.meanReturns.index, columns=['allocation'])
        minVol_allocation.allocation = [round(i*100,0) for i in minVol_allocation.allocation]

        # Efficient Frontier
        efficientAllocation = []
        efficientList = []
        targetReturns = np.linspace(minVol_returns, maxSR_returns, iterations)
        # Value of minimization problem
        for target in targetReturns:
            efficientList.append(self._efficientOpt(self.meanReturns, self.covMatrix, target, constraintSet)['fun'])
        # Value of allocations
        for target in targetReturns:
            x = self._efficientOpt(self.meanReturns, self.covMatrix, target, constraintSet)['x']
            x = [float('{:.6f}'.format(val)) for val in x]
            efficientAllocation.append(x)


        maxSR_returns, maxSR_std = round(maxSR_returns*100,2), round(maxSR_std*100,2)
        minVol_returns, minVol_std = round(minVol_returns*100,2), round(minVol_std*100,2)


        df = pd.DataFrame([efficientList, targetReturns, efficientAllocation]).T
        df.columns = ['Volatility', 'Return', 'Allocations']
        return df


    # Visuazing the Efficient Frontier
    '''
    Plot the efficient frontier of allocations.
    '''
    def plotEfficientFrontier(self, iterations, riskFreeRate=0, constraintSet=(0,1)):
        '''
        Return a graph ploting the min vol, max sr and efficient frontier.
        '''
        '''
        Set 13 week treasury bill rate as free-risk ratio,
        from the same dates as the selected stocks.
        '''
        if riskFreeRate == '13-week':
            riskFreeRate = self.getRiskFreeRate()
        

        df = self.efficientFrontier(iterations, riskFreeRate, constraintSet)

        #Efficient Frontier
        EF_curve = go.Scatter(
            name='Efficient Frontier',
            mode='lines',
            x=[round(ef_std*100, 2) for ef_std in df['Volatility']],
            y=[round(target*100, 2) for target in df['Return']],
            line=dict(color='black', width=3, dash='dashdot')
        )

        #Max SR
        MaxSharpeRatio = go.Scatter(
            name='Maximium Sharpe Ratio',
            mode='markers',
            x=[round(df['Volatility'].iat[-1]*100,2)],
            y=[round(df['Return'].iat[-1]*100,2)],
            marker=dict(color='red',size=14,line=dict(width=2, color='black'))
        )

        #Min Vol
        MinVol = go.Scatter(
            name='Mininium Volatility',
            mode='markers',
            x=[round(df['Volatility'].iat[0]*100,2)],
            y=[round(df['Return'].iat[0]*100,2)],
            marker=dict(color='green',size=14,line=dict(width=2, color='black'))
        )

        data = [EF_curve, MaxSharpeRatio, MinVol]

        layout = go.Layout(
            title = 'Portfolio Optimisation with the Efficient Frontier',
            yaxis = dict(title='Annualised return (%)'),
            xaxis = dict(title='Annualised volatility (%)'),
            showlegend = True,
            legend = dict(
                x = 0.75, y = 0, traceorder='normal',
                bgcolor='#E2E2E2',
                bordercolor='black',
                borderwidth=2),
            width=800,
            height=600)
        
        fig = go.Figure(data=data, layout=layout)
        customdata=np.stack((df['Volatility'].apply(lambda x: round(x * 100, 2)),
                             df['Return'].apply(lambda x: round(x * 100, 2)),
                             df['Allocations'].apply(lambda x: [round(val*100, 2) for val in x])),
                             axis=1)
        fig.update_traces(customdata=customdata, hovertemplate=('Volatility: %{customdata[0]}<br>' + 
                                                                'Return: %{customdata[1]}<br>' + 
                                                                'Allocations: %{customdata[2]} <br>' + 
                                                                '<extra></extra>'))
        
        return fig.show()