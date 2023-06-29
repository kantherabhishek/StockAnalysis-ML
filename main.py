import dash
from dash import dcc, html, dash_table
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime
from yahoofinancials import YahooFinancials
import numpy_financial as npf

# Create a Dash application
app = dash.Dash(__name__)

# Define the layout for the Dash application
app.layout = html.Div([
    html.Div([
        html.H1("Fundamental and Technical Stock Analaysis using Machine Learning"),
        dcc.Input(id="symbol-input", type="text", value="AAPL", placeholder="Enter symbol",
                  style={"width": "50%", "padding": "12px 20px", "margin": "8px 0", "box-sizing": "border-box",
                         "border": "3px solid #555"}),
        dcc.DatePickerRange(
            id='date-range',
            start_date='2019-01-01',
            end_date='2023-01-01',
            start_date_placeholder_text="Start Date",
            end_date_placeholder_text="End Date",
            style={"width": "50%", "padding": "12px 20px", "margin": "8px 0", "box-sizing": "border-box",
                "border": "3px solid #555"}
        )
    ], style={"display": "flex", "flexDirection": "column"}),

        dcc.Input(id="discount-rate-input", type="number", value=0.035, placeholder="Enter discount rate. For 4.76% use 0.0475",
                  style={"width": "50%", "padding": "12px 20px", "margin": "8px 0", "box-sizing": "border-box",
                         "border": "3px solid #555"}),

    html.Div([
        html.Button(id="submit-button", n_clicks=0, children="Submit",
                    style={"width": "50%", "fontSize": "20px", "backgroundColor": "#04AA6D", "border": "none",
                           "color": "white", "padding": "16px 32px", "textDecoration": "none",
                           "margin": "4px 2px", "cursor": "pointer"})
    ], style={"margin": "10px 0"}),
    dcc.Tabs(id="tabs", value="tab-technical", children=[
        dcc.Tab(label="Technical Analysis", value="tab-technical", children=[
            html.H1("Technical Analaysis using Machine Learning"),
            html.Div(id="technical-analysis-content")
        ]),
        dcc.Tab(label="Financial Analysis", value="tab-financial", children=[
            # Placeholder for financial analysis content
            html.Div(id="financial-analysis-content")
        ])
    ]),
    dcc.Graph(id="chart")
])


@app.callback(
    [
        dash.dependencies.Output("technical-analysis-content", "children"),
        dash.dependencies.Output("chart", "figure"),
        dash.dependencies.Output("financial-analysis-content", "children")
    ],
    [
        dash.dependencies.Input("submit-button", "n_clicks")
    ],
    [
        dash.dependencies.State("symbol-input", "value"),
        dash.dependencies.State("date-range", "start_date"),
        dash.dependencies.State("date-range", "end_date"),
        dash.dependencies.State("discount-rate-input", "value")

    ]
)

def update_company_info(n_clicks, symbol, start_date, end_date, discount_rate):
    yahoo_financials = YahooFinancials(symbol)
    balance_sheet_data_qt = yahoo_financials.get_financial_stmts('quarterly', 'balance')
    balance_sheet_data_qt_df = pd.DataFrame(balance_sheet_data_qt['balanceSheetHistoryQuarterly'][symbol])
    # Convert DataFrame to HTML table
    balance_sheet_data_qt_df_table = html.Div(
        [
            html.Div(create_table(balance_sheet_data_qt_df.iloc[i][col]), style={'display': 'inline-block'})
            for i in range(len(balance_sheet_data_qt_df))
            for col in balance_sheet_data_qt_df.columns
            if not pd.isnull(balance_sheet_data_qt_df.iloc[i][col])
        ]
    )

    income_statement_data_qt = yahoo_financials.get_financial_stmts('quarterly', 'income')
    income_statement_table = pd.DataFrame(income_statement_data_qt['incomeStatementHistoryQuarterly'][symbol])
    income_statement_table_qt_df_table = html.Div(
        [
            html.Div(create_table(income_statement_table.iloc[i][col]), style={'display': 'inline-block'})
            for i in range(len(income_statement_table))
            for col in income_statement_table.columns
            if not pd.isnull(income_statement_table.iloc[i][col])
        ]
    )


    all_statement_data_qt = yahoo_financials.get_financial_stmts('quarterly', ['income', 'cash', 'balance'])
    all_statement_table = pd.DataFrame(all_statement_data_qt['incomeStatementHistoryQuarterly'][symbol])
    all_statement_table_qt_df_table = html.Div(
        [
            html.Div(create_table(all_statement_table.iloc[i][col]), style={'display': 'inline-block'})
            for i in range(len(all_statement_table))
            for col in all_statement_table.columns
            if not pd.isnull(all_statement_table.iloc[i][col])
        ]
    )
    # Set pandas display options to show all columns and rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    # Discount rate (used for NPV and IRR calculations)
    discount_rate = discount_rate
    stock_data = yf.download(symbol, start=start_date, end=end_date, group_by='ticker')
    adj_close_prices = stock_data['Adj Close'].tolist()
    monthly_returns = pd.Series(adj_close_prices).pct_change().dropna()

    # Calculate NPV and IRR
    cash_flows = [0] * 11
    cash_flows[0] = -1  # initial investment
    for i in range(1, 11):
        cash_flows[i] = monthly_returns.iloc[i-1] * 100  # hypothetical monthly investment of $100
    npv = npf.npv(discount_rate, cash_flows)
    irr = npf.irr(cash_flows)

     # Create the fundamental analysis content
    content = html.Div([
        html.H3("NPV and IRR Calculations"),
        html.H1(f"Discount Rate: {round(discount_rate*100,2)} %"),
        html.H1(f"NPV: {round(npv,4)}"),
        html.H1(f"IRR: {round(irr*100,2)} %"),
        dash.html.H2("Financial Data"),
        dash.html.H3("Balance Sheet Data (Last 5 Quarterly) Starting"),
        dash.html.H4(datetime.strptime(str(list(balance_sheet_data_qt['balanceSheetHistoryQuarterly'][symbol][0].keys())[0]), "%Y-%m-%d").date()),
        dash.html.Div((balance_sheet_data_qt_df_table)),
        dash.html.H3("Income Statement Data (Last 5 Quarterly) Starting"),
        dash.html.H4(datetime.strptime(str(list(income_statement_data_qt['incomeStatementHistoryQuarterly'][symbol][0].keys())[0]), "%Y-%m-%d").date()),
        dash.html.Div(income_statement_table_qt_df_table),
        dash.html.H3("All Statement Data Starting"),
        dash.html.H4(datetime.strptime(str(list(all_statement_data_qt['incomeStatementHistoryQuarterly'][symbol][0].keys())[0]), "%Y-%m-%d").date()),
        dash.html.Div(all_statement_table_qt_df_table),
    
        
    ])


    # Fetch data for the ticker from Yahoo Finance
    data = yf.Ticker(symbol).history(start=start_date, end=end_date)
    company_info = yf.Ticker(symbol).info

    # Determine train and test years based on the data length
    data_length = len(data)
    train_years = int(data_length * 0.7)  # 70% of data for training
    test_years = data_length - train_years

    # Splitting data into train and test sets
    train_data = data[:train_years]
    test_data = data[train_years:]

    # Fitting the SARIMAX model
    model_sarimax = SARIMAX(train_data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results_sarimax = model_sarimax.fit()

    # Forecasting with SARIMAX
    forecast_sarimax = results_sarimax.get_forecast(steps=len(test_data))
    forecast_ci_sarimax = forecast_sarimax.conf_int()

    # Creating pandas dataframe to plot the predicted values from SARIMAX
    test_pred_sarimax = pd.DataFrame(forecast_sarimax.predicted_mean.values, index=test_data.index, columns=['SARIMAX'])

    # Preparing data for AdaBoost
    X_train = np.arange(len(train_data)).reshape(-1, 1)
    y_train = train_data['Close'].values

    X_test = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)

    # Fitting the AdaBoost model
    model_ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=100)
    model_ada.fit(X_train, y_train)

    # Forecasting with AdaBoost
    forecast_ada = model_ada.predict(X_test)

    # Creating pandas dataframe to plot the predicted values from AdaBoost
    test_pred_ada = pd.DataFrame(forecast_ada, index=test_data.index, columns=['AdaBoost'])
    ex_dividend_date = company_info.get("exDividendDate")
    if ex_dividend_date:
        ex_dividend_date = datetime.utcfromtimestamp(ex_dividend_date).strftime('%d-%M-%Y')



    # Create technical analysis content
    technical_analysis_content = html.Table([
        html.Tr([html.Th("Previous Close"), html.Td(company_info.get("previousClose"))]),
        html.Tr([html.Th("Open"), html.Td(company_info.get("open"))]),
        html.Tr([html.Th("Bid"), html.Td(company_info.get("bid"))]),
        html.Tr([html.Th("Ask"), html.Td(company_info.get("ask"))]),
        html.Tr([html.Th("Day's Range"), html.Td(str(company_info.get("dayLow")) + " - " + str(company_info.get("dayHigh")))]),
        html.Tr([html.Th("52 Week Range"), html.Td(str(company_info.get("fiftyTwoWeekLow")) + " - " + str(company_info.get("fiftyTwoWeekHigh")))]),
        html.Tr([html.Th("Volume"), html.Td(company_info.get("volume"))]),
        html.Tr([html.Th("Avg. Volume"), html.Td(company_info.get("averageVolume"))]),
        html.Tr([html.Th("Market Cap"), html.Td(company_info.get("marketCap"))]),
        html.Tr([html.Th("Beta (5Y Monthly)"), html.Td(company_info.get("beta"))]),
        html.Tr([html.Th("PE Ratio (TTM)"), html.Td(company_info.get("trailingPE"))]),
        html.Tr([html.Th("EPS (TTM)"), html.Td(company_info.get("trailingEps"))]),
        html.Tr([html.Th("Forward Dividend & Yield"), html.Td(str(company_info.get("dividendRate")) + " - " + str(company_info.get("dividendYield")))]),
        html.Tr([html.Th("Ex-Dividend Date"), html.Td(ex_dividend_date)])
        
    ], style={"border-collapse": "collapse", 
              "width": "100%",
              "padding": "8px",
              "text-align": "left",
              "border-bottom": "3px solid #ddd"
                        })

    # Create chart figure
    chart_figure = {
        "data": [
            {"x": data.index, "y": data["Open"], "type": "line", "name": "Open"},
            {"x": data.index, "y": data["Close"], "type": "line", "name": "Adjusted Close"},
            {"x": data.index, "y": data["Volume"], "type": "bar", "name": "Volume", "yaxis": "y2"},
            {"x": test_pred_sarimax.index, "y": test_pred_sarimax["SARIMAX"], "type": "line", "name": "SARIMAX"},
            {"x": test_pred_ada.index, "y": test_pred_ada["AdaBoost"], "type": "line", "name": "AdaBoost"}
        ],
        "layout": {
            "title": f"{symbol} Stock Prices",
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "Price"},
            "yaxis2": {"title": "Volume", "overlaying": "y", "side": "right"}
        }
    }
    

    return technical_analysis_content, chart_figure, content
def create_table(data):
    # Convert data to DataFrame and add an index
    df = pd.DataFrame([data], columns=data.keys())
    
    # Transpose the data
    transposed_data = df.transpose().reset_index()
    transposed_data.columns = ["Particulars", 'Value']
    
    # Convert transposed data to HTML table
    table = dash_table.DataTable(
        data=transposed_data.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in transposed_data.columns],
        style_cell={'textAlign': 'left'},
        style_header={'fontWeight': 'bold'}
    )

    return table

# Run the Dash application
if __name__ == "__main__":
    app.run_server(debug=True)
