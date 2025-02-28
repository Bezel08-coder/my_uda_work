

import pandas as pd

# Step 1: Load portfolio holdings CSV

# Read the CSV into a DataFrame
portfolio_df = pd.read_csv("my_portfolio4.csv")


# Preview the data
print("Portfolio Data:")
print(portfolio_df.head(10))

import requests
import pandas as pd
from datetime import datetime, timedelta

# Define the FRED API key
fred_api_key = "10d8f98c2565d70748dbe9dcb1020eb3"

# Define the endpoint and parameters
fred_url = "https://api.stlouisfed.org/fred/series/observations"

# Calculate the start date as 60 TRADING days ago (about ~84 calendar days to account for weekends/holidays)
end_date = datetime.today().strftime("%Y-%m-%d")  # Today's date
start_date = (datetime.today() - timedelta(days=84)).strftime("%Y-%m-%d")  # 60 trading days ago

# Request DGS3MO (daily 3-month T-Bill yield)
params = {
    "series_id": "DGS3MO",  # Daily 3-Month Treasury Rate
    "api_key": fred_api_key,
    "file_type": "json",
    "observation_start": start_date,  # Fetch ~84 days to ensure 60 trading days
    "observation_end": end_date,
}

# Make the request
response = requests.get(fred_url, params=params)
data = response.json()

# Extract observations
observations = data.get("observations", [])

# Convert to DataFrame
t_bill_df = pd.DataFrame(observations)[["date", "value"]]
t_bill_df["value"] = pd.to_numeric(t_bill_df["value"], errors="coerce")  # Convert to numeric
t_bill_df.rename(columns={"value": "DGS3MO"}, inplace=True)  # Rename column for clarity

# Keep only the most recent 60 trading days
t_bill_df = t_bill_df.tail(60)

# Check if there are any NaN values in the 'TB3MS' column
if t_bill_df['DGS3MO'].isna().any():
    print("NaN values found, filling with previous day's rate")
    t_bill_df['DGS3MO'] = t_bill_df['DGS3MO'].fillna(method='ffill')
else:
    print("No NaN values found in 'DGS3MO' column")


# Save the data locally
t_bill_df.to_csv("DGS3MO_data.csv", index=False)

# Display the first few rows
print(t_bill_df.head())


import requests
import pandas as pd
from datetime import datetime, timedelta

# Define Alpha Vantage API key
alpha_vantage_api_key = "3NK7KIM6Q5HBL54V"



# Get unique tickers from portfolio and add SPY (ETF tracking the S&P 500)
tickers = list(portfolio_df["Ticker"].unique())
tickers.append("SPY")

# Define time frame: last 60 trading days (~84 calendar days to account for weekends/holidays)
end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=84)).strftime("%Y-%m-%d")

# Function to fetch daily close prices (using TIME_SERIES_DAILY)
def fetch_alpha_vantage_data(ticker):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "apikey": alpha_vantage_api_key,
        "outputsize": "compact",
        "datatype": "json",
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if "Time Series (Daily)" in data:
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df = df.rename(columns={"4. close": ticker})
        df = df[[ticker]].astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df.loc[start_date:end_date]
    else:
        print(f"Error fetching data for {ticker}: {data}")
        return pd.DataFrame()

# Fetch data for each ticker and store in a list of DataFrames
price_dfs = [fetch_alpha_vantage_data(ticker) for ticker in tickers]

# Merge all price data on the date index; use dropna() to keep only common dates
price_data = pd.concat(price_dfs, axis=1).dropna().sort_index()

# Save the merged data to CSV
price_data.to_csv("portfolio_price_data.csv", index=True)

# Display the first few rows of the merged DataFrame
print(price_data.head())



# Reset the index in price_data so that the date becomes a column
price_data_reset = price_data.reset_index().rename(columns={"index": "date"})
price_data_reset["date"] = pd.to_datetime(price_data_reset["date"])
t_bill_df["date"] = pd.to_datetime(t_bill_df["date"])


# Merge price_data_reset and t_bill_df on the "date" column using a left join
merged_df = pd.merge(price_data_reset, t_bill_df, on="date", how="left")

# Sort the merged DataFrame by date
merged_df.sort_values("date", inplace=True)

merged_df["DGS3MO"] = merged_df["DGS3MO"].fillna(method='ffill')

# Display the first few rows of the merged DataFrame
print(merged_df.head())

merged_df.to_csv("merged_df.csv", index=False)
print("Merged DataFrame saved as 'merged_df.csv'.")

# Define the tickers to exclude from this process
exclude_tickers = {"SPY", "DGS3MO", "date"}

# Get unique tickers from portfolio_df (they should be consistent across dates)
unique_tickers = portfolio_df["Ticker"].unique()

# Loop through each unique ticker
for ticker in unique_tickers:
    # Process only if the ticker is in merged_df and not in the exclusion list
    if ticker in merged_df.columns and ticker not in exclude_tickers:
        # Get the constant Units and AssetClass values from portfolio_df for this ticker
        units = portfolio_df.loc[portfolio_df["Ticker"] == ticker, "Units"].unique()[0]
        asset_class = portfolio_df.loc[portfolio_df["Ticker"] == ticker, "AssetClass"].unique()[0]
        
        # Create new columns in merged_df for this ticker
        merged_df[ticker + "_Units"] = units
        merged_df[ticker + "_Asset_Class"] = asset_class

# Display the head of the final DataFrame
final_df = merged_df.copy()

final_df.to_csv("final_df.csv", index=False)
print("final_df saved as 'final_df.csv'.")

print(final_df.head())



# Loop over columns to calculate returns for eligible columns
for col in final_df.columns:
    if col != "date" and col != "DGS3MO" and "Units" not in col and "Asset_Class" not in col:
        final_df[col + "_Returns"] = final_df[col].pct_change()

# Display the first few rows to check the new returns columns
print(final_df.head())


# Initialize a new Series to accumulate the portfolio value for each row
portfolio_value = pd.Series(0, index=final_df.index)

# Loop over each column in final_df
for col in final_df.columns:
    # Check if the column is a candidate for a ticker price column
    if col not in ["date", "SPY", "DGS3MO"] and "Returns" not in col and "Asset_Class" not in col and not col.endswith("_Units"):
        units_col = col + "_Units"  # Corresponding units column
        if units_col in final_df.columns:
            # Multiply the price by the units and add to the portfolio_value
            portfolio_value += final_df[col] * final_df[units_col]

# Assign the computed series as a new column in final_df
final_df["Portfolio_Value"] = portfolio_value

# Display the updated DataFrame
print(final_df.head())



final_df["Portfolio_Returns"] = final_df["Portfolio_Value"].pct_change()


final_df["Portfolio_Cumulative_Returns"] = (1 + final_df["Portfolio_Returns"]).cumprod() - 1
print(final_df[["date", "Portfolio_Returns", "Portfolio_Cumulative_Returns"]].head())


# Initialize asset class total columns with zeros
final_df["Equity"] = 0.0
final_df["Fixed_Income"] = 0.0
final_df["Alternative_Asset"] = 0.0

# Define columns to exclude
excluded_cols = {"date", "SPY", "DGS3MO", "Portfolio_Value"}

# Loop over each column in final_df
for col in final_df.columns:
    # Process only columns that are not excluded and do not end with specified suffixes
    if (col not in excluded_cols and 
        not col.endswith("_Returns") and 
        not col.endswith("_Units") and 
        not col.endswith("_Asset_Class")):
        
        # Expected corresponding units and asset class columns for this ticker
        units_col = col + "_Units"
        asset_class_col = col + "_Asset_Class"
        
        # Check if both corresponding columns exist
        if units_col in final_df.columns and asset_class_col in final_df.columns:
            # Get the asset class from the first row (assuming it's constant)
            asset_class = final_df[asset_class_col].iloc[0]
            # Calculate the value by multiplying the price column with its units
            value_series = final_df[col] * final_df[units_col]
            
            # Add the computed value to the appropriate asset class column
            if asset_class == "Equity":
                final_df["Equity"] += value_series
            elif asset_class == "Fixed Income":
                final_df["Fixed_Income"] += value_series
            elif asset_class == "Alternative Asset":
                final_df["Alternative_Asset"] += value_series


final_df.to_csv("final_df.csv", index=False)
print("final_df saved as 'final_df.csv'.")

# Display a preview of the updated DataFrame
print(final_df.head())



portfolio_volatility = final_df["Portfolio_Returns"].iloc[1:].std()
print("Portfolio Volatility:", f"{portfolio_volatility:.2%}")

import pandas as pd

# Identify returns columns for portfolio tickers
return_cols = [col for col in final_df.columns 
               if col.endswith("_Returns") 
               and col not in ["SPY_Returns","Portfolio_Returns", "Portfolio_Cumulative_Returns"]]

# Extract ticker names by removing the "_Returns" suffix
tickers_returns = [col.replace("_Returns", "") for col in return_cols]

# Get the last row of final_df (the latest date)
last_row = final_df.iloc[-1]

# Compute weights from the last date for each ticker
last_date_weights = {}
for ticker in tickers_returns:
    price_col = ticker        # e.g., "AAPL"
    units_col = ticker + "_Units"  # e.g., "AAPL_Units"
    if price_col in final_df.columns and units_col in final_df.columns:
        # Weight = (Price × Units) / Portfolio_Value (for the last date)
        last_date_weights[ticker] = (last_row[price_col] * last_row[units_col]) / last_row["Portfolio_Value"]

# Compute the correlation matrix for the returns columns
corr_matrix = final_df[return_cols].corr()

# Compute weighted average correlation for all unique ticker pairs using last date weights
weighted_sum = 0.0
total_weight_product = 0.0
n = len(tickers_returns)
for i in range(n):
    for j in range(i+1, n):
        ticker_i = tickers_returns[i]
        ticker_j = tickers_returns[j]
        # Product of the last date's weights for ticker_i and ticker_j
        weight_product = last_date_weights[ticker_i] * last_date_weights[ticker_j]
        # Retrieve the pairwise correlation from the correlation matrix
        corr_value = corr_matrix.loc[ticker_i + "_Returns", ticker_j + "_Returns"]
        weighted_sum += corr_value * weight_product
        total_weight_product += weight_product

portfolio_correlation = weighted_sum / total_weight_product if total_weight_product != 0 else None

print("Weighted Average Portfolio Correlation (using last date weights):", portfolio_correlation)



import numpy as np

# Convert the annualized T‑Bill rate to a daily rate (assuming 252 trading days)
risk_free_daily = final_df["DGS3MO"] / 100 / 252

# Compute daily excess returns: Portfolio_Returns minus the daily risk-free rate
daily_excess = final_df["Portfolio_Returns"] - risk_free_daily

# Calculate the mean and standard deviation of the daily excess returns, ignoring NaN values
mean_excess_daily = daily_excess.mean()
std_excess_daily = daily_excess.std()

# Annualize the daily average excess return and the volatility
annualized_mean_excess = mean_excess_daily * 252
annualized_std = std_excess_daily * np.sqrt(252)

# Compute the Sharpe Ratio
sharpe_ratio = annualized_mean_excess / annualized_std

print("Portfolio Sharpe Ratio:", sharpe_ratio)


import statsmodels.api as sm

# Assume merged_df has columns "Portfolio_Returns" and "SPY_Returns" (the benchmark)
returns_df = final_df[["Portfolio_Returns", "SPY_Returns"]].dropna()

# Set up the regression: market returns as the independent variable
X = returns_df["SPY_Returns"]
X = sm.add_constant(X)  # adds an intercept term

# Portfolio returns as the dependent variable
y = returns_df["Portfolio_Returns"]

model = sm.OLS(y, X).fit()
portfolio_beta = model.params["SPY_Returns"]

print("Portfolio Beta:", portfolio_beta)



import numpy as np

# Convert DGS3MO (annual percentage) to a daily risk-free rate
final_df["daily_rf"] = final_df["DGS3MO"].apply(lambda x: x / 100 / 252)

# Calculate daily excess returns for the portfolio and the market
daily_excess_portfolio = final_df["Portfolio_Returns"] - final_df["daily_rf"]
daily_excess_market = final_df["SPY_Returns"] - final_df["daily_rf"]

# Compute the mean daily excess returns
avg_excess_portfolio = daily_excess_portfolio.mean()
avg_excess_market = daily_excess_market.mean()

# Calculate daily Jensen's alpha as the difference between the portfolio's average excess return
# and the product of portfolio beta and the market's average excess return
daily_jensen_alpha = avg_excess_portfolio - portfolio_beta * avg_excess_market

# Annualize Jensen's alpha by multiplying by 252 (approximate number of trading days per year)
jensen_alpha_annualized = daily_jensen_alpha * 252

print("Annualized Jensen's Alpha:", f"{jensen_alpha_annualized:.2%}")


import streamlit as st
import pandas as pd
import altair as alt
from streamlit_autorefresh import st_autorefresh

# -------------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------------------
# AUTO-REFRESH (Refresh every 24 hours, i.e., 86,400,000 ms)
# -------------------------------------------------------------------------
st_autorefresh(interval=86400000, limit=1, key="daily_refresh")




portfolio_value_today = final_df["Portfolio_Value"].iloc[-1]
daily_pl = final_df["Portfolio_Returns"].iloc[-1]


# -------------------------------------------------------------------------
# DASHBOARD TITLE (Centered, normal size)
# -------------------------------------------------------------------------
st.markdown(
    "<h2 style='text-align: center;'>Portfolio Dashboard</h2>",
    unsafe_allow_html=True
)

# -------------------------------------------------------------------------
# TOP ROW: ASSET ALLOCATION (LEFT) & PORTFOLIO VALUE (RIGHT)
# -------------------------------------------------------------------------
col1, col2 = st.columns(2)

# 1) ASSET ALLOCATION (PIE CHART) -----------------------------------------
latest_date = final_df["date"].max()
latest_data = final_df.loc[final_df["date"] == latest_date].iloc[0]

alloc_df = pd.DataFrame({
    "AssetClass": ["Equity", "Fixed Income", "Alternative Asset"],
    "Value": [latest_data["Equity"], latest_data["Fixed_Income"], latest_data["Alternative_Asset"]]
})

# defining a color scale using a light orange, plus other non-red colors
color_scale = alt.Scale(
    domain=["Equity", "Fixed Income", "Alternative Asset"],
    range=["#42A5F5", "#F57C00", "#2ca02c"]  
)

pie_chart = (
    alt.Chart(alloc_df)
    .mark_arc(outerRadius=140, innerRadius=70)  
    .encode(
        theta=alt.Theta("Value:Q"),
        color=alt.Color("AssetClass:N", scale=color_scale),
        tooltip=[alt.Tooltip("AssetClass:N"), alt.Tooltip("Value:Q", format=",.2f")]
    )
    .properties(
        width=350,
        height=350,
        title="Asset Allocation"
    )
    # configuring the chart title to use Trebuchet MS, size ~20, centered
    .configure_title(
        anchor="middle",
        font="Trebuchet MS",
        fontSize=20
    )
)

col1.altair_chart(pie_chart, use_container_width=True)

# 2) PORTFOLIO VALUE OVER TIME (AREA CHART) -------------------------------
value_chart = (
    alt.Chart(final_df)
    .mark_area(
        line={"color": "#1f77b4"},
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="#87CEEB", offset=0),
                alt.GradientStop(color="#FFFFFF", offset=1)
            ]
        )
    )
    .encode(
        x=alt.X("date:T", axis=alt.Axis(format="%b %d, %Y", title="Date", grid=False)),
        y=alt.Y("Portfolio_Value:Q", axis=alt.Axis(format="$,.0f", title="Value ($)", grid=False)),
        tooltip=[alt.Tooltip("Portfolio_Value:Q", format="$,.0f"), alt.Tooltip("date:T")]
    )
    .properties(
        width=600,
        height=350,
        title="Portfolio Value"
    )
    .configure_title(
        anchor="middle",
        font="Trebuchet MS",
        fontSize=20
    )
)

col2.altair_chart(value_chart, use_container_width=True)

# -------------------------------------------------------------------------
# BOTTOM ROW: METRICS (Portfolio Value, Daily P/L, Correlation, Volatility, Sharpe, Jensen's Alpha)
# -------------------------------------------------------------------------
mcols = st.columns(6)

# Helper to display a big metric with an arrow
def big_number(col, heading, value_str, actual_value, invert=False):
    # invert=False: positive => green up, negative => red down 
    # invert=True: positive => red up, negative => green down 
    if actual_value >= 0:
        arrow_html = "<span style='font-size:24px; color:green;'>▲</span>" if not invert else "<span style='font-size:24px; color:red;'>▲</span>"
    else:
        arrow_html = "<span style='font-size:24px; color:red;'>▼</span>" if not invert else "<span style='font-size:24px; color:green;'>▼</span>"
    
    col.markdown(f"""
        <div style='text-align:center;'>
            <h4 style='margin-bottom:0px; font-family:"Trebuchet MS"; font-size:20px;'>{heading}</h4>
            <p style='font-size:20px; margin-top:0px; font-family:"Trebuchet MS";'>
                {value_str} {arrow_html}
            </p>
        </div>
    """, unsafe_allow_html=True)


# displaying metrics using big_number helper
big_number(mcols[0], "Portfolio Value", f"${portfolio_value_today:,.0f}", portfolio_value_today, invert=False)

big_number(mcols[1], "Daily P/L", f"{daily_pl:.2%}", daily_pl, invert=False)
big_number(mcols[2], "Correlation", f"{portfolio_correlation:.2f}", portfolio_correlation, invert=True)
big_number(mcols[3], "Volatility", f"{portfolio_volatility:.2%}", portfolio_volatility, invert=True)
big_number(mcols[4], "Sharpe Ratio", f"{sharpe_ratio:.2f}", sharpe_ratio, invert=False)
big_number(mcols[5], "Jensen's Alpha", f"{jensen_alpha_annualized:.2%}", jensen_alpha_annualized, invert=False)

# -------------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------------
st.markdown("<p style='text-align:center; font-size:12px;'>Dashboard auto-refreshes every 24 hours</p>", unsafe_allow_html=True)
