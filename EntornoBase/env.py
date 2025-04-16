import os
import csv
import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from gymnasium.spaces import Discrete, Box
from collections import namedtuple, deque
from copy import deepcopy
import math
import time
import torch
import torch.nn.functional as F
from tabulate import tabulate
import pandas as pd
import yfinance as yf
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("La versión de Gym utilizada en esta PEC es la 0.28.1, compruébalo a continuación")
print("Gym Version:", gym.__version__)  # 1.0.0
print("Torch Version:", torch.__version__)  # 2.5.1+cu121

class StockMarketEnv(gym.Env):
    def __init__(self, initial_balance=10000,  is_eval = False,
                  save_to_csv=False, csv_filename="stock_trading_log.csv", data_file = "preprocessed_data_1d.csv"):
        super(StockMarketEnv, self).__init__()

        # Descargar los datos históricos de la acción
        self.df = pd.read_csv(data_file)
        self.num_trading_days = len(self.df)
        self.prices = self.df['adjcp'].values.squeeze()
        self.n_steps = len(self.prices)-1

        # Parámetros del entorno
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance

        self.previus_net_worth = initial_balance

        # Espacio de acciones: 0 -> mantener, 1 -> comprar, 2 -> vender
        self.action_space = Discrete(3)

        # Calculamos los indicadores técnicos
        self.rsi = self.df['rsi']
        self.macd = self.df['macd']
        self.cci = self.df['cci']
        self.adx = self.df['adx']
        self.turbulence = self.df['turbulence']



        # Espacio de observaciones: [precio_actual, balance, acciones, rsi, macd, cci, adx, turbulence]
        self.observation_space = Box(low=0, high=np.inf, shape=(8,), dtype=np.float32)
        self.is_eval = is_eval

        # Valores para normalización (obtenemos mínimos y máximos)
        self.min_price = np.min(self.prices)
        self.max_price = np.max(self.prices)

        self.min_rsi = np.min(self.rsi)
        self.max_rsi = np.max(self.rsi)

        self.min_macd = np.min(self.macd)
        self.max_macd = np.max(self.macd)
        
        self.min_cci = np.min(self.cci)
        self.max_cci = np.max(self.cci)
        
        self.min_adx = np.min(self.adx)
        self.max_adx = np.max(self.adx)
        
        self.min_turbulence = np.min(self.turbulence)
        self.max_turbulence = np.max(self.turbulence)


        # Parámetros adicionales para el CSV
        self.save_to_csv = save_to_csv
        self.csv_filename = csv_filename

        # Si la opción de almacenar en CSV está activada, crea o sobreescribe el archivo
        if self.save_to_csv:
            if os.path.exists(self.csv_filename):
                os.remove(self.csv_filename)
            with open(self.csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Step", "Balance", "Shares Held", "Net Worth", "Profit"])


    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
            
        self.balance = self.initial_balance
        self.previus_net_worth = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = 0
        return self._next_observation(),{}

    def _normalize(self, value, min_val, max_val):
        normalized_value = (value - min_val) / (max_val - min_val) if max_val != min_val else 0
        # Verifica si el resultado es un array o lista con un solo elemento
        if isinstance(normalized_value, (np.ndarray, list)) and len(normalized_value) == 1:
            # Si es un array o lista de un solo valor, devuelve ese valor escalar
            return normalized_value[0]

        # De lo contrario, devuelve el valor normalizado (puede ser escalar o array)
        return normalized_value

    def _next_observation(self):
        # Normalizamos los valores
        norm_price = self._normalize(self.prices[self.current_step], self.min_price, self.max_price)
        norm_balance = self._normalize(self.balance, self.initial_balance * 0.85, self.initial_balance * 1.25)
        norm_shares_held = self._normalize(self.shares_held, 0, 100)  # Máximo de 100 acciones
        norm_rsi = self._normalize(self.rsi[self.current_step], self.min_rsi, self.max_rsi)
        norm_macd = self._normalize(self.macd[self.current_step], self.min_macd, self.max_macd)
        norm_cci = self._normalize(self.cci[self.current_step], self.min_cci, self.max_cci)
        norm_adx = self._normalize(self.adx[self.current_step], self.min_adx, self.max_adx)
        norm_turbulence = self._normalize(self.turbulence[self.current_step], self.min_turbulence, self.max_turbulence)

        return np.array([
            norm_price,
            norm_balance,
            norm_shares_held,
            norm_rsi,
            norm_macd,
            norm_cci,
            norm_adx,
            norm_turbulence
        ])


    def step(self, action):
        current_price = self.prices[self.current_step]
        reward = 0

        # Acción: 0 -> mantener, 1 -> comprar, 2 -> vender
        if action == 0 and self.shares_held== 0 and self.current_step > 0:
            if  self.prices[self.current_step] > self.prices[self.current_step-1]:
                reward = -1
            else:
                reward = 1

        else:
            if action == 1:  # Comprar
                while self.balance >= current_price:
                    self.shares_held += 1
                    self.balance -= current_price

            elif action == 2:  # Vender
                while self.shares_held > 0:
                    self.shares_held -= 1
                    self.balance += current_price

            # Actualizar el patrimonio neto
            self.net_worth = self.balance + self.shares_held * current_price

            # Recompensa: patrimonio neto anterior y actual.
            reward = 1 if self.net_worth > self.previus_net_worth else (0 if self.net_worth == self.previus_net_worth else -1)


        # Actualizar el precio anterior
        self.previus_net_worth = self.net_worth


        # Avanzar al siguiente paso
        self.current_step += 1
        terminated = self.current_step >= self.n_steps
        truncated = self.net_worth < self.initial_balance* 0.85

        if self.save_to_csv:
            self.save_to_csv_file()



        # Devuelve la observación, la recompensa, si está hecho, y otra info adicional
        return self._next_observation(), reward, terminated, truncated, {'profit': self.net_worth - self.initial_balance}

    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Net worth: {self.net_worth}')
        print(f'Profit: {profit}')



    # La función save_to_csv_file guarda los datos actuales en un archivo CSV.
    # 1. Primero calcula el beneficio como la diferencia entre el valor neto
    # actual y el balance inicial.
    # 2. Luego, abre (o crea) el archivo CSV en modo 'append' para agregar una
    # nueva fila de datos sin sobrescribir las anteriores.
    # 3. Escribe una nueva fila en el CSV con los valores del paso actual,
    # balance, acciones mantenidas, valor neto y el beneficio.
    # Step,Balance,Shares Held,Net Worth,Profit
    # 1,12000,50,13000,3000
    def save_to_csv_file(self):
        """Guarda los datos actuales en el archivo CSV."""
        profit = self.net_worth - self.initial_balance
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.current_step, self.balance, self.shares_held, self.net_worth, profit])

