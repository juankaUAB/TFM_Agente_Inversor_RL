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

class StockMarketEnv(gym.Env):
    def __init__(self, initial_balance=10000,  is_eval = False,
                  save_to_csv=False, csv_filename="stock_trading_log.csv", data_file = "preprocessed_data_1d.csv",
                  window_size = 6, step = 'train'):
        super(StockMarketEnv, self).__init__()

        # Descargar los datos históricos de la acción
        prov_df = pd.read_csv(data_file)
        percent = int(len(prov_df) * 0.7)
        if step == 'train':
            self.df = prov_df[:percent]
        elif step == 'test':
            self.df = prov_df[percent:]
        else:
            raise Exception
        self.num_trading_days = len(self.df)
        self.prices = self.df['adjcp'].values.squeeze()
        self.n_steps = len(self.prices)-1

        # Parámetros del entorno
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.current_step = self.window_size
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance

        self.previus_net_worth = initial_balance

        # Espacio de acciones continuo: < 0 fracción de venta, > 1 fracción de compra
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Calculamos los indicadores técnicos
        self.rsi = self.df['rsi']
        self.macd = self.df['macd']
        self.cci = self.df['cci']
        self.adx = self.df['adx']
        self.turbulence = self.df['turbulence']



        # Espacio de observaciones: window_size * [precio_actual, balance, acciones, rsi, macd, cci, adx, turbulence]
        self.observation_space = Box(low=0, high=np.inf, shape=(self.window_size * 8,), dtype=np.float32)
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
        if seed is None:
            super().reset(seed=42)
        else:
            super().reset(seed=seed)
            
        self.balance = self.initial_balance
        self.previus_net_worth = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = self.window_size
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
        obs = []
    
        for i in range(self.current_step - self.window_size, self.current_step):
            norm_price = self._normalize(self.prices[i], self.min_price, self.max_price)
            norm_balance = self._normalize(self.balance, self.initial_balance * 0.85, self.initial_balance * 1.25)
            norm_shares_held = self._normalize(self.shares_held, 0, 100)
            norm_rsi = self._normalize(self.rsi[i], self.min_rsi, self.max_rsi)
            norm_macd = self._normalize(self.macd[i], self.min_macd, self.max_macd)
            norm_cci = self._normalize(self.cci[i], self.min_cci, self.max_cci)
            norm_adx = self._normalize(self.adx[i], self.min_adx, self.max_adx)
            norm_turbulence = self._normalize(self.turbulence[i], self.min_turbulence, self.max_turbulence)
    
            obs.extend([
                norm_price,
                norm_balance,
                norm_shares_held,
                norm_rsi,
                norm_macd,
                norm_cci,
                norm_adx,
                norm_turbulence
            ])

        return np.array(obs, dtype=np.float32)


    def step(self, action):
        
        current_price = self.prices[self.current_step]
        reward = 0
        factor_multiplicativo_reward = 1
        
        action = action[0]

        # Acción: -1 -> vender, 0 -> mantener, 1 -> comprar
        if action > 0:  # Comprar
            if self.rsi[self.current_step] < 30:
                factor_multiplicativo_reward += 0.05  # Pequeña recompensa por tomar una acción de compra en un RSI bajo.
            if self.current_step > 0 and self.macd[self.current_step] > self.macd[self.current_step - 1]:
                factor_multiplicativo_reward += 0.02 # MACD cruzando arriba
            amount_to_buy = int(self.balance * action / current_price)  # Usa el valor de acción para determinar cuánto comprar
            if amount_to_buy > 0:
                self.shares_held += amount_to_buy
                self.balance -= amount_to_buy * current_price
            else:
                factor_multiplicativo_reward -= 0.05 # Penalización por intentar comprar sin balance disponible

        elif action < 0:  # Vender
            if self.rsi[self.current_step] > 70:
                factor_multiplicativo_reward += 0.05  # Fuerte recompensa por vender cuando el RSI indica sobrecompra
            if self.current_step > 0 and self.macd[self.current_step] > self.macd[self.current_step - 1]:
                factor_multiplicativo_reward += 0.02 # MACD cruzando abajo
            if self.current_step >= 5 and self.prices[self.current_step] > max(self.prices[self.current_step-5:self.current_step+5]):  # Vender en el punto máximo, precio más alto en los últimos 10 días.
                factor_multiplicativo_reward += 0.20
            amount_to_sell = int(self.shares_held * -action)  # Usa el valor negativo de la acción para determinar cuánto vender
            if amount_to_sell > 0:
                self.shares_held -= amount_to_sell
                self.balance += amount_to_sell * current_price
            else:
                factor_multiplicativo_reward -= 0.05 # Penalización por intentar vender sin acciones
        
        if (action != 0) and (self.turbulence[self.current_step] > 0.8):  # Si decide comprar o vender en un mercado volátil
            factor_multiplicativo_reward -= 0.02  # Penalización por riesgo

        # Actualizar el patrimonio neto
        self.net_worth = self.balance + self.shares_held * current_price
        
        if self.net_worth < self.initial_balance * 0.9:
            factor_multiplicativo_reward -= 0.05  # Penalización moderada por grandes pérdidas
        if self.net_worth > self.previus_net_worth * 1.2:
            factor_multiplicativo_reward += 0.05  # Recompensa leve por mantener un crecimiento constante
        
        if action == 0 and self.current_step > 0:
            if self.shares_held == 0:
                if self.prices[self.current_step] > self.prices[self.current_step - 1]:
                    # El precio sube y el agente no compró
                    factor_multiplicativo_reward -= 0.02
                elif self.prices[self.current_step] < self.prices[self.current_step - 1]:
                    # El precio baja y no compró
                    factor_multiplicativo_reward += 0.02
            else:
                if self.prices[self.current_step] > self.prices[self.current_step - 1]:
                    # Precio sube y mantiene
                    factor_multiplicativo_reward += 0.02
                elif self.prices[self.current_step] < self.prices[self.current_step - 1]:
                    # Precio baja y no vende
                    factor_multiplicativo_reward -= 0.02
                
            if self.net_worth > self.previus_net_worth:
                factor_multiplicativo_reward += 0.05  # Si el balance neto aumenta
            elif self.net_worth < self.previus_net_worth:
                factor_multiplicativo_reward -= 0.05  # Si el balance neto disminuye
        
        
        factor_multiplicativo_reward = np.clip(factor_multiplicativo_reward, 0.5, 2)  # Establecer límites para evitar cambios drásticos
        long_term_reward = self.net_worth - self.initial_balance  # Recompensa acumulada desde el inicio
        reward = (self.net_worth - self.previus_net_worth) * factor_multiplicativo_reward
        
        #En caso de recompensa 0 y se deba aplicar penalización o recompensa adicional se aplicará sobre el precio actual
        if reward == 0 and factor_multiplicativo_reward != 1:
            reward = -(current_price + (current_price * (1 - factor_multiplicativo_reward))) if factor_multiplicativo_reward < 1 else current_price * factor_multiplicativo_reward
        
        reward = reward + (long_term_reward / 1000)


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

