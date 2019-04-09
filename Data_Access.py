import quandl
import pandas as pd

# Function extracting news information from Apple
def get_apple_news():
    # Adding API key to access Apple News Table
    quandl.ApiConfig.api_key = 'L56d_fbyzwLzgtWyc2Zx'
    df = quandl.get_table('IFT/NSA', ticker='AAPL', exchange_cd='US')

    # Formatting dataframe and dropping unnecessary columns
    df = df.sort_values(by='date')
    df.index = df.date
    df.drop(['ticker'], 1, inplace=True)
    df.drop(['exchange_cd'], 1, inplace=True)
    df.drop(['name'], 1, inplace=True)
    df.drop(['date'], 1, inplace=True)

    return df

# Function extracting stock information from Apple
def get_apple_stock():
    df = pd.read_csv('Data/EOD-AAPL.csv')
    # Reverses index order
    df = df.reindex(index=df.index[::-1])

    # Sets date as index
    df.index = df.Date

    # Using adjusted values instead of nominal due to stock splits and dividend payments
    df['Open'] = df['Adj_Open']
    df['High'] = df['Adj_High']
    df['Low'] = df['Adj_Low']
    df['Close'] = df['Adj_Close']
    df['Volume'] = df['Adj_Volume']

    # Dropping unnecessary columns
    df.drop(['Date'], 1, inplace=True)
    df.drop(['Split'], 1, inplace=True)
    df.drop(['Adj_Open'], 1, inplace=True)
    df.drop(['Adj_High'], 1, inplace=True)
    df.drop(['Adj_Low'], 1, inplace=True)
    df.drop(['Adj_Close'], 1, inplace=True)
    df.drop(['Adj_Volume'], 1, inplace=True)

    # Extracting new features related to movement of stock
    df['Delta_Abs'] = (df['Close'] - df['Open'])
    df['Delta_Rel'] = df['Delta_Abs'] / df['Open']

    return df

# Function extracting genera news
def get_general_news():
    df = pd.read_csv('Data/General_News.csv')
    df.index = df.Date
    df.drop(['Label'], 1, inplace=True)
    df.drop(['Date'], 1, inplace=True)

    return df

# Function extracting index information from DJIA
def get_DJIA_index():
    df = pd.read_csv('Data/DJIA_table.csv')
    df = df.reindex(index=df.index[::-1])  # Reverses index order
    df["Adj Close"] = df.Close  # Moving close to the last column
    df.index = df.Date
    df.drop(['Date'], 1, inplace=True)
    df.drop(['Close'], 1, inplace=True)

    return df
