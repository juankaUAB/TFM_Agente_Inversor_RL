from datetime import date
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import pytz
from dateutil.relativedelta import relativedelta

def descargar_datos_1d():
    # Leer los tickers desde un archivo CSV
    df = pd.read_csv('tickers.csv', header=None, encoding='utf-8')
    tickers = df[0].to_list()

    # Calcular el rango de fechas de los últimos 60 días
    ed = date.today()
    sd = ed - relativedelta(years=16)

    # Convertir las fechas en objetos datetime con zona horaria específica
    tz = pytz.timezone('America/New_York')
    ed_tz = tz.localize(pd.Timestamp(ed))
    sd_tz = tz.localize(pd.Timestamp(sd))

    # Inicializar el DataFrame resultante
    resultado = pd.DataFrame()

    # Configurar el número de tickers por archivo CSV
    tickers_por_archivo = 100
    archivo_actual = 1

    # Descargar los datos de cada ticker
    for i, ticker in enumerate(tqdm(tickers, desc="Descargando datos"), start=1):
        try:
            datos = yf.download(tickers=ticker, start=sd_tz, end=ed_tz, interval="1d")
            datos['tic'] = ticker
            datos['datadate'] = datos.index
            resultado = pd.concat([resultado, datos], axis=0)
        except Exception as e:
            print(f"Error al descargar datos para {ticker}: {str(e)}")
            continue

        # Guardar los datos en un archivo CSV y reiniciar el DataFrame si se alcanza el límite de tickers por archivo
        if i % tickers_por_archivo == 0 or i == len(tickers):
            resultado.to_csv(f'datos_1d_{archivo_actual}.csv', index=False)
            resultado = pd.DataFrame()
            archivo_actual += 1




descargar_datos_1d()