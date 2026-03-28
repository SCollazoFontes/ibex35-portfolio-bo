import numpy as np


def sharpe_ratio(pesos, rendimientos, matriz_covarianzas, risk_free_rate):
    """
    Calcula el Ratio de Sharpe para una cartera dada.
    Devuelve el valor negativo para convertirlo en problema de minimización.
    """
    rendimiento_cartera = np.dot(pesos, rendimientos)
    riesgo_cartera      = np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianzas, pesos)))
    return -(rendimiento_cartera - risk_free_rate) / riesgo_cartera
