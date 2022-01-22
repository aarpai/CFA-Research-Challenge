# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 18:15:33 2021

@author: Nicola
"""

import financialmodelingprep_data_import as statements
import alphavantage_data_import as alphav
import financialEvaluation as val
# import relativevaluation as rv

import pandas as pd
import os
import numpy as np

with open('API_KEYS.txt', 'r') as f:
    api_keys = f.read().splitlines() # [financialmodelingprep, alphavantage]
    
ticker = 'HLI'

# Functions
def earnings_forecast(years, payout_ratio, initial_equity, re, roe, cost_of_equity):
    '''Forecast Equity, Net Income, Equity Cost and Excess Return for a given number of years - follows Aswath Damodaran theory'''
    n = np.arange(1, years)
    equity_forecasts = np.zeros(years)
    payout_forecasts = np.ones(years)*payout_ratio
    roe_forecasts = np.ones(years)*roe
    net_income_forecasts = np.zeros(years)
    equity_cost = np.zeros(years)
        
    equity_forecasts[0] = initial_equity
    net_income_forecasts[0] = initial_equity * roe
    equity_cost[0] = equity_forecasts[0] * cost_of_equity
    for i in n:
        re = net_income_forecasts[i-1] * (1-payout_forecasts[i-1])
        equity_forecasts[i] = equity_forecasts[i-1] + re
        net_income_forecasts[i] = equity_forecasts[i] * roe_forecasts[i]
        equity_cost[i] = equity_forecasts[i] * cost_of_equity
    excess_returns = np.subtract(net_income_forecasts, equity_cost)
    return equity_forecasts, net_income_forecasts, equity_cost, excess_returns


# Body
if __name__ == "__main__":
    if os.path.exists('pd_statements/income_statement') and os.path.exists('pd_statements/balance_sheet') and os.path.exists('pd_statements/dividends_history'):
        income_statement = pd.read_pickle('pd_statements/income_statement')
        balance_sheet = pd.read_pickle('pd_statements/balance_sheet')
        dividends = pd.read_pickle('pd_statements/dividends_history')
    
    else:        
        income_statement = statements.income_statement(ticker, api_keys[0])
        balance_sheet = statements.balance_sheet(ticker, api_keys[0])
        dividends = statements.dividends_history(ticker, api_keys[0])
        print('DB Queried')
    
        try:
            os.makedirs('pd_statements/')
        except:
            pass
        income_statement.to_pickle('pd_statements/income_statement')
        balance_sheet.to_pickle('pd_statements/balance_sheet')
        dividends.to_pickle('pd_statements/dividends_history')
    
    annual_dividends = dividends[['adjDividend', 'dividend']].groupby(by=[dividends.index.year], sort=False).sum() # US based firms pay quarterly dividends. We get the annual dividend summing the 4 most recent
    annual_dividends.index = pd.to_datetime(annual_dividends.index, format='%Y') + pd.offsets.MonthBegin(2) + pd.offsets.Day(30) # offset dates to allow operations with financial statements dataframes
    ## finalDividend = initialDividend * (1 + annualDividendGrowth)**years
    #annual_dividend_growth = (annual_dividends.iloc[0]['adjDividend'] / annual_dividends.iloc[5]['adjDividend'])**(1/6) - 1
    
    # Change in Equity statements are not available in free APIs
    change_in_equity = pd.DataFrame(columns = ['Dividends', 'OtherSharesRepurchased/Forfeited'], index = income_statement.index) # No considerable amount of debt has been issued to buy back shares
    change_in_equity.iloc[0] = [89464000, 120036000]
    change_in_equity.iloc[1] = [82790000, 61119000]
    change_in_equity.iloc[2] = [70415000, 34975000 + 36535000]
    change_in_equity.iloc[3] = [51305000, 15139000 + 36352000]
    change_in_equity.iloc[4] = [47883000, 27309000]
    annual_dividend_growth = (change_in_equity['Dividends'][0]/income_statement.iloc[0]['weightedAverageShsOut'] / (change_in_equity['Dividends'][-1]/income_statement.iloc[-1]['weightedAverageShsOut']))**(1/5) - 1
    
    
    ## Assumptions
    
    valutation_weights = [0.3, 0.1, 0.6] # excess return estimation, DDM estimation, relative valuation estimation weights
    rf = 0.06/100 # 1-month rate - Longstaff - https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/textview.aspx?data=yield
    rf = float(alphav.get_treasury_yields(api_keys[1], 'monthly', '10year').iloc[0])/100
    beta = 1.21 # 'Competitors_earnings_by_segment.csv' average of narrowly defined industry beta
    mpremium = 4.72/100 # current implied premium S&P 500 – January 1, 2021 https://deliverypdf.ssrn.com/delivery.php?ID=853127084071024085001030017102065077046084047001025025024007112112122001028097084126052032042044047025048106014097012120075097000058035046014106071016121089076033041018104022021080087031003071009122113110094114029026111118024121112118092102093118&EXT=pdf&INDEX=TRUE
    #mpremium = 4.38/100 # https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3879109, bibliography
    sustainable_growth_rate = 3/100 # approximately the growth rate of the economy between 2010-2020
    cost_of_equity = val.capm(rf, beta, mpremium)
    
    ### Gordon Growth Model
    ## roe = net income / initial shareholders equity --we use initial equity instead of avg to make ROE an appropriate measure of return
    accounting_ratios = pd.DataFrame(income_statement['netIncome']/balance_sheet['totalStockholdersEquity'].shift(-1), columns=['ROE'])
    
    ## g = ROE(1 - payout ratio)
    accounting_ratios['Payout Ratio'] = change_in_equity['Dividends'][:len(income_statement['eps'])] / income_statement['netIncome']
    accounting_ratios['Modified Payout Ratio'] = ((change_in_equity['Dividends'][:len(income_statement['eps'])] + change_in_equity['OtherSharesRepurchased/Forfeited'][:len(income_statement['eps'])].mean()) / income_statement['netIncome']) # modified to include share buybacks contribution
    accounting_ratios['g - Sustainable Growth Rate'] = accounting_ratios['ROE'] * (1 - accounting_ratios['Payout Ratio'])
    accounting_ratios['Modified g - Sustainable Growth Rate'] = accounting_ratios['ROE'] * (1 - accounting_ratios['Modified Payout Ratio'])
    # we get a very high g from HLI statements. It is therefore wiser to get a reference g from the industry and estimate a meaningful payout ratio
    
    ## During a recession (temporary) the values of net income and other accounting
    ## measure may be severily affected and not represent a meaningful
    ## base of estimation for future years earnings
    ## therefore it is reasonable to exclude the year and consider
    ## an average of the ROE in previous 5-10 years
    ##
    ## To forecast next period earnings:
    ## E = book value of equity * normalized ROE
    ## We can now estimate the dividend payout ratio:
    ## payout ratio = 1 - g / normalized ROE
    ## And using the gordon growth model:
    ## Intrinsic Value of Equity = E * payout ratio / (cost of equity - g)
    ## Our estimations are ROE and cost of equity ->
    ## we should check value estimation when these parameters change
    
    normalized_roe = accounting_ratios['ROE'][1:-1].mean() # excluding last year
    normalized_roe_assumption = normalized_roe
    
    ## Forecast profits and Equity for the next 5 years
    ytd = [balance_sheet['totalStockholdersEquity'][0],\
                               accounting_ratios['ROE'][0],\
                                   np.nan,\
                                       income_statement['netIncome'][0],\
                                           income_statement['netIncome'][0]*cost_of_equity, \
                                               np.nan, np.nan, np.nan,\
                                               accounting_ratios['Modified Payout Ratio'][0],\
                                                   np.nan,\
                                                       income_statement['netIncome'][0]*(1-accounting_ratios['Modified Payout Ratio'][0])]
    
    ytd[5] = ytd[3] - ytd[-1]
    periods = 4
    cat = ['BeginningBVofEquity', 'ROE', 'CostofEquity', 'NetIncome', 'EquityCost', \
           'ExcessEquityReturn', 'CumulatedCostofEquity', 'PresentValue', \
               'DividendPayoutRatio', 'DividendsPaid', 'RetainedEarnings']
    excess_returns_table = pd.DataFrame(columns=cat, index=np.arange(periods))
    excess_returns_table.iloc[0] = ytd
    excess_returns_table['ROE'] = normalized_roe_assumption
    excess_returns_table['DividendPayoutRatio'] = accounting_ratios['Modified Payout Ratio'].mean()
    
    excess_returns_table['BeginningBVofEquity'][1:], excess_returns_table['NetIncome'][1:], excess_returns_table['EquityCost'][1:], excess_returns_table['ExcessEquityReturn'][1:] = \
        earnings_forecast(periods-1, excess_returns_table['DividendPayoutRatio'][0], ytd[0], ytd[-1], normalized_roe_assumption, cost_of_equity)
            
    excess_returns_table['DividendsPaid'] = excess_returns_table['NetIncome']*excess_returns_table['DividendPayoutRatio']
    excess_returns_table['RetainedEarnings'] = excess_returns_table['NetIncome'] - excess_returns_table['DividendsPaid']
    excess_returns_table['CostofEquity'] = cost_of_equity
    excess_returns_table['CumulatedCostofEquity'][1:] = [x**(i+1) for i,x in enumerate(1+excess_returns_table['CostofEquity'][1:]*np.ones(periods-1))]
    excess_returns_table['PresentValue'] = excess_returns_table['ExcessEquityReturn'] / excess_returns_table['CumulatedCostofEquity']

    ## From year 6 we assume a period of stable growth
    ## In stable growth we assume that the present value of excess returns is equal to zero
    intrinsic_value = (excess_returns_table['PresentValue'].sum() + balance_sheet['totalStockholdersEquity'][0]) / income_statement.iloc[0]['weightedAverageShsOut']
    
    ## If we assume that in stable growth there is still a positive excess return
    beta = 1
    cost_of_equity = val.capm(rf, beta, mpremium)
    stable_roe = sustainable_growth_rate/(1-accounting_ratios['Modified Payout Ratio'].mean()) #0.09 # temporary - look for industry avg
    next_period_earnings = excess_returns_table['BeginningBVofEquity'].iloc[-1] * stable_roe * (1+sustainable_growth_rate)
    terminal_value_of_excess_return = next_period_earnings / (cost_of_equity - sustainable_growth_rate)
    present_value_of_excess_return = terminal_value_of_excess_return / excess_returns_table['CumulatedCostofEquity'].iloc[-1]
    intrinsic_value = intrinsic_value + present_value_of_excess_return / income_statement.iloc[0]['weightedAverageShsOut']
    
    weighted_value = valutation_weights[0] * intrinsic_value
    
    # scenarios = ['Case1', 'Case2', 'Case3']
    # intrinsic_value_table = pd.DataFrame(columns=['Next Period EPS', 'ROE', 'Modified Payout Ratio', 'Cost of Equity', 'Estimated Value'], index=scenarios)   
    
    # for i in range(len(scenarios)):
    #     next_period_earnings = balance_sheet['totalStockholdersEquity'][0]/income_statement.iloc[0]['weightedAverageShsOut'] * normalized_roe
    #     estimated_payout_ratio = 1 - sustainable_growth_rate / accounting_ratios['ROE'][1:-1].mean()
    #     intrinsic_value = next_period_earnings*estimated_payout_ratio / (cost_of_equity - sustainable_growth_rate)
        
    #     intrinsic_value_table.loc[scenarios[i]] = [next_period_earnings, normalized_roe, estimated_payout_ratio, cost_of_equity, intrinsic_value]
        
    #     cost_of_equity += 0.01
    #     normalized_roe -= 0.05
    
    
    ## Share buybacks positively affect shareholders but, unlike dividends, they are not consistent over time. Hence we consider the average over the years
    cost_of_equity = val.capm(rf, beta, mpremium)
    avg_yearly_buybacks = change_in_equity['OtherSharesRepurchased/Forfeited'].mean()
    cash_flows_to_shareholders = (change_in_equity['Dividends'][0] + avg_yearly_buybacks) / income_statement.iloc[0]['weightedAverageShsOut']
    unstable_growth_rate = (excess_returns_table['ROE'] * (1 - accounting_ratios['Modified Payout Ratio'].mean()))[0]
    intrinsic_value = val.gordon_growth(cash_flows_to_shareholders, cost_of_equity, sustainable_growth_rate, unstable_growth_rate, 3)
    # (current_dividend, cost_of_capital, stable_growth_rate, unstable_growth_rate=0, unstable_growth_period=0)
    weighted_value += valutation_weights[1] * intrinsic_value
    
    
    ### Multiplier Models
    ## primary competitors vary by product and industry expertise and would include the following:
    ## for our CF practice, Jefferies LLC,Lazard Ltd, Moelis & Company, N M Rothschild & Sons Limited, Piper Sandler Companies, Robert W. Baird & Co. Incorporated, StifelFinancial Corp., William Blair & Company, L.L.C., and the bulge-bracket investment banking firms
    ## for our FR practice, Evercore Partners,Lazard Ltd, Moelis & Company, N M Rothschild & Sons Limited and PJT Partners
    ## for our FVA practice, Duff & Phelps Corp., the “bigfour” accounting firms, and various global financial advisory firms
    
    ## Question: Should I incorporate more companies in the list of consider only the peers
    competitors = ['HLI', 'JEF', 'LAZ', 'MC', 'PIPR', 'SF', 'EVR', 'GHL', 'JPM', 'GS', 'BAC', 'MS', 'C', 'UBS', 'CS', 'DB', 'HSBC', 'BCS', 'RY', 'WFC', 'NMR', 'BMO', 'MUFG'] # ! remove PJT P/B > 12x        
    
    # pb_table, corr_matrix, predicted_pb, predicted_value = rv.priceToBookComparison(api_keys, ticker, accounting_ratios['ROE'][0], balance_sheet['totalStockholdersEquity'][0], income_statement.iloc[0]['weightedAverageShsOut'], competitors)
    
    # weighted_value += valutation_weights[2] * predicted_value
    
    
    
## Information by Segment and Geography
    # revenue_by_segment = cl.read_csv_statements('pd_statements/revenue_by_segment.csv')
    # revenue_by_segment.to_pickle('pd_statements/revenue_by_segment')
    
    # revenue_by_region = cl.read_csv_statements('pd_statements/revenue_by_region.csv')
    # revenue_by_region.to_pickle('pd_statements/revenue_by_region')
    
    # income_by_region = cl.read_csv_statements('pd_statements/income_by_region.csv')
    # income_by_region.to_pickle('pd_statements/income_by_region')
    
    # assets_by_region = cl.read_csv_statements('pd_statements/assets_by_region.csv')
    # assets_by_region.to_pickle('pd_statements/assets_by_region')
    
    # assets_by_segment = cl.read_csv_statements('pd_statements/assets_by_segment.csv')
    # assets_by_segment.to_pickle('pd_statements/assets_by_segment')