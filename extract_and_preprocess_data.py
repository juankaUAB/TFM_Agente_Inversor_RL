import pandas as pd
import numpy as np
from stockstats import StockDataFrame as Sdf
import ta

def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(file_name)
    _data= _data.drop_duplicates()
    return _data

def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data=data.sort_values(['datadate','tic'],ignore_index=True)
    #data  = data[final_columns]
    data.index = data.datadate.factorize()[0]
    return data

def calcualte_price(df):
    """
    calcualte adjusted close price, open-high-low price and volume
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    data = df.copy()
    data = data[['datadate', 'tic', 'Close'#, 'ajexdi'
                 , 'Open', 'High', 'Low', 'Volume']]
    #data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)



    data['adjcp'] = data['Close']  #/ data['ajexdi']
    data['open'] = data['Open'] #/ data['ajexdi']
    data['high'] = data['High'] #/ data['ajexdi']
    data['low'] = data['Low'] #/ data['ajexdi']
    data['volume'] = data['Volume']

    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data

def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())

    stock['close'] = pd.to_numeric(stock['adjcp'])
    stock['low'] = pd.to_numeric(stock['low'])
    stock['high'] = pd.to_numeric(stock['high'])
    unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    #temp = stock[stock.tic == unique_ticker[0]]['macd']
    for i in range(len(unique_ticker)):
        
        ## macd
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = pd.concat([macd,temp_macd], ignore_index=True)
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = pd.concat([rsi,temp_rsi], ignore_index=True)
        ## cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = pd.concat([cci,temp_cci], ignore_index=True)
        ## adx
        temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = pd.concat([dx,temp_dx], ignore_index=True)

    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx

    return df



def preprocess_data(file):
    """data preprocessing pipeline"""

    df = load_dataset(file_name=file)[1:]
    # get data after 2009
    #df = df[df.datadate>=20090000]
    # calcualte adjusted price
    df_preprocess = calcualte_price(df)
    # add technical indicators using stockstats
    df_final=add_technical_indicator(df_preprocess)
    # fill the missing values at the beginning
    df_final.fillna(method='bfill',inplace=True)
    return df_final

def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    df[['adjcp', 'open', 'high', 'low', 'volume']] = df[['adjcp', 'open', 'high', 'low', 'volume']].apply(pd.to_numeric)
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate','tic']).reset_index(drop=True)
    return df



def calcualte_turbulence(df):
    """calculate turbulence index based on dow 30"""
    # can add other market assets
    
    df_price_pivot=df.pivot(index='datadate', columns='tic', values='adjcp')
    unique_date = df.datadate.unique()
    # start after a year
    start = 252
    turbulence_index = [0]*start
    #turbulence_index = [0]
    count=0
    for i in range(start,len(unique_date)):
        print(i)
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        cov_temp = hist_price.cov()
        current_temp=(current_price - np.mean(hist_price,axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp>0:
            count+=1
            if count>2:
                turbulence_temp = temp[0][0]
            else:
                #avoid large outlier because of the calculation just begins
                turbulence_temp=0
        else:
            turbulence_temp=0
        turbulence_index.append(turbulence_temp)
    
    
    turbulence_index = pd.DataFrame({'datadate':df_price_pivot.index,
                                     'turbulence':turbulence_index})
    return turbulence_index

csv_files_1d = [
"datos_1d_1.csv",
]


dataframes_1d = []
dataframes_5min = []

for file in csv_files_1d:
    file_path = f"{file}"
    preprocessed_data = preprocess_data(file_path)
    preprocessed_data_with_turbulence = add_turbulence(preprocessed_data)
    dataframes_1d.append(preprocessed_data_with_turbulence)


final_dataframe_1d = pd.concat(dataframes_1d, ignore_index=True)

final_dataframe_1d.to_csv("preprocessed_data_1d.csv", index=False)