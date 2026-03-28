"""
Función objetivo: ratio de Sharpe para la cartera.
"""
import numpy as np


def sharpe_ratio(pesos, rendimientos, matriz_covarianzas, risk_free_rate):
    """
    Calcula el ratio de Sharpe para una cartera.
    Se devuelve el negativo para plantear minimización.
    """
    rendimiento_cartera = np.dot(pesos, rendimientos)
    riesgo_cartera = np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianzas, pesos)))
    sr = (rendimiento_cartera - risk_free_rate) / riesgo_cartera
    return -sr
