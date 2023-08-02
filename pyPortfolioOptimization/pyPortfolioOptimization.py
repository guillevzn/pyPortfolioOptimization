import numpy as np
import pandas as pd
import scipy as sc
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as go
import random

class pyPortfolioOptimization:

    def __init__(self, stocks, start, end):
        yf.pdr_override()
        self.stocks = stocks
        self.start = start
        self.end = end
        self.called_getData = False


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
        if self.called_getData == True:
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
        if self.called_getData == True:
            pass
        else:
            stockData = yf.download(self.stocks, start=self.start, end=self.end)
            self.stockData = stockData
            self.len_period = len(stockData)
            stockData = stockData['Close']

            returns = stockData.pct_change()
            meanReturns = returns.mean()
            covMatrix = returns.cov()

            self.meanReturns = meanReturns
            self.covMatrix = covMatrix

            self.called_getData = True

        return meanReturns, covMatrix


    # Plot data
    '''
    Plot downloaded data.
    '''
    def plotData(self, figsize=[None, None]):
        self.check_getData()
        
        pd.options.plotting.backend = "plotly"

        fig = go.Figure(px.line(self.stockData['Close'],
                                custom_data=['variable']))

        fig.update_layout(
            title='Close values for stocks',
            yaxis = dict(title='Daily value'),
            xaxis = dict(title='Date (days)'),
            showlegend = True,
            legend = dict(
                x = 1.005, y = 0, traceorder='normal',
                bgcolor='#E2E2E2',
                bordercolor='black',
                borderwidth=2),
            legend_title_text = '<b>Tickers</b>',
            width=figsize[0],
            height=figsize[1]
        )

        fig.update_traces(hovertemplate=('Ticker: %{customdata[0]}<br>' +
                                         'Date: %{x}<br>' +
                                         'Price: %{y:.2f}' +
                                         '<extra></extra>'))
        
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
    def efficientFrontier(self, iterations, riskFreeRate=0, constraintSet=(0,1), separated_allocations=False):
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

        '''
        '''

        maxSR_returns, maxSR_std = round(maxSR_returns*100,2), round(maxSR_std*100,2)
        minVol_returns, minVol_std = round(minVol_returns*100,2), round(minVol_std*100,2)


        df = pd.DataFrame([efficientList, targetReturns, efficientAllocation]).T
        df.columns = ['Volatility', 'Return', 'Allocations']

        if separated_allocations==True:
            expanded_allocations = df.apply(lambda row: pd.Series(row['Allocations'], index=self.stockData['Close'].columns.to_list()), axis=1)
            result_df = pd.concat([df[['Volatility', 'Return']], expanded_allocations], axis=1)
            return result_df
        else:
            return df


    # Visuazing the Efficient Frontier
    '''
    Plot the efficient frontier of allocations.
    '''
    def plotEfficientFrontier(self, iterations, riskFreeRate=0, constraintSet=(0,1), figsize=[None, None]):
        '''
        Return a graph ploting the min vol, max sr and efficient frontier.
        '''
        '''
        Set 13 week treasury bill rate as free-risk ratio,
        from the same dates as the selected stocks.
        '''
        if riskFreeRate == '13-week':
            riskFreeRate = self.getRiskFreeRate()


        df = self.efficientFrontier(iterations, riskFreeRate, constraintSet, separated_allocations=True)

        df = df.apply(lambda x: [round(val * 100, 2) for val in x])

        customdata=np.stack(([df[column].values for column in df]),
                             axis=1)
        
        # Create the 'Allocations' part of the hovertemplate dynamically for each column
        allocation_template = '<br>'.join([f'{col}: %{{customdata[{i+2}]}}%' for i, col in enumerate(df.columns[2:])])


        # Generate a list of random hexadecimal color codes
        color_dict = {col: '#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for col in df.columns[2:]}


        fig = make_subplots(rows=2,
                            cols=2,
                            specs=[[{"colspan": 2}, None], [{}, {}]],
                            subplot_titles=('Efficient Frontier', 'Evolution of Allocations: Volatility', 'Evolution of Allocations: Return')
        )

        #Efficient Frontier
        fig.add_trace(go.Scatter(
            x=df['Volatility'],
            y=df['Return'],
            name='Efficient Frontier',
            mode='lines+markers',
            marker=dict(color='black',size=5),
            showlegend=False),
            row=1, col=1
        )
        
        #Evolution of Allocations Volatility
        for col in df.columns[2:]:
            fig.add_trace(go.Scatter(x=df['Volatility'], y=df[col].values,
                                     name = col,
                                     mode = 'lines',
                                     legendgroup=col,
                                     showlegend=True,
                                     marker=dict(color=color_dict[col])),
                                     row=2, col=1)
        
        #Evolution of Allocations Return
        for col in df.columns[2:]:
            fig.add_trace(go.Scatter(x=df['Return'], y=df[col].values,
                                     name = col,
                                     mode = 'lines',
                                     legendgroup=col,
                                     showlegend=False,
                                     marker=dict(color=color_dict[col])),
                                     row=2, col=2
        )
        
        # Minimum Volatility
        fig.add_annotation(showarrow=False,
                   arrowhead=0,
                   align = 'left',
                   x=df['Volatility'].min(),
                   y=df['Return'].min(),
                   text="Mininium Volatility",
                   opacity=0.7,
                   yshift=-15
        )
        
        # Maximum Shaper Ratio
        fig.add_annotation(showarrow=False,
                   arrowhead=0,
                   align = 'right',
                   x=df['Volatility'].max(),
                   y=df['Return'].max(),
                   text="Maximium Sharpe Ratio",
                   opacity=0.7,
                   yshift=15
        )
        
        # Subplot 1
        fig.update_xaxes(
            title_text='Return (%)',
            row=1,
            col=1
        )

        fig.update_yaxes(
            title_text='Volatility (%)',
            row=1,
            col=1
        )

        # Subplot 2
        fig.update_xaxes(
            title_text='Volatility (%)',
            showspikes=True,
            spikecolor="green",
            spikesnap="cursor",
            spikemode="across",
            spikedash="solid",
            row=2,
            col=1
        )

        fig.update_yaxes(
            title_text='Allocation (%)',
            showspikes=True,
            spikecolor="green",
            spikesnap="cursor",
            spikemode="across",
            spikedash="solid",
            row=2,
            col=1
        )
        
        # Subplot 3
        fig.update_xaxes(
            title_text='Return (%)',
            showspikes=True,
            spikecolor="green",
            spikesnap="cursor",
            spikemode="across",
            spikedash="solid",
            row=2,
            col=2
        )

        fig.update_yaxes(
            title_text='Allocation (%)',
            showspikes=True,
            spikecolor="green",
            spikesnap="cursor",
            spikemode="across",
            spikedash="solid",
            row=2,
            col=2
        )

        fig.update_traces(customdata=customdata, hovertemplate=('Return: %{x}%<br>' +
                                                                'Volatility: %{y}%<br>' +
                                                                '<br>' +
                                                                f'Allocations<br>{allocation_template}<br>' +
                                                                '<extra></extra>')
        )
        
        return fig.show()