# ==============================================================================
# FARC-HD (Fuzzy Association Rule-based Classification Model for High-Dimensional)
# ==============================================================================
# Original Algorithm Design & Java Implementation:
# @author Jesus Alcalá-Fdez (University of Granada) - 09/02/2010
# @copyright KEEL Copyright (c) 2008-2010
#
# Python Translation, Scikit-learn API Integration & Optimization:
# @author Iñaki Mellado Ilundain
# @author JOSE ANTONIO SANZ DELGADO
#
# Description: 
#   This file is part of the Python port of the FARC-HD algorithm, 
#   originally developed for the KEEL software tool. It has been redesigned 
#   to be fully compatible with the scikit-learn ecosystem and optimized 
#   using Numba for high-performance JIT compilation.
# ==============================================================================

import numpy as np
from numba import njit,prange

# -------------------------------------------------------------------------
# NOMBRE: fuzzificacion_parcial
# DESCRIPCIÓN: 
#   Calcula el grado de pertenencia de forma paralela para un conjunto de etiquetas.
# ENTRADA:
#   - valores [np.array]: Matriz de índices (variable, índice local, columna).
#   - data_matrix [np.array]: Datos brutos del dataset.
#   - x0_arr, x1_arr, x3_arr [np.array]: Puntos definitorios de los triángulos.
#   - trainFuzzificado [np.array]: Matriz de salida para almacenar resultados.
# -------------------------------------------------------------------------
@njit(fastmath=True, parallel=True)
def fuzzificacion_parcial(valores, data_matrix, x0_arr, x1_arr, x3_arr, trainFuzzificado):
    n_labels = valores.shape[0]
    n_data = data_matrix.shape[0]

    # Bucle paralelo por etiquetas
    for l in prange(n_labels):
        if valores[l, 2] != -1:
            var_idx = valores[l, 0]
            local_idx = valores[l, 1]
            col_idx = valores[l, 2]

            # 1. Extraemos los vértices del triángulo (Nombres claros)
            a = x0_arr[var_idx, local_idx] # Inicio
            b = x1_arr[var_idx, local_idx] # Pico
            c = x3_arr[var_idx, local_idx] # Fin

            # 2. Pre-calculamos los denominadores para mantener la velocidad
            # (Multiplicar es más rápido que dividir dentro del bucle)
            ancho_izq = b - a
            ancho_der = c - b
            
            inv_izq = 1.0 / ancho_izq if ancho_izq != 0 else 0.0
            inv_der = 1.0 / ancho_der if ancho_der != 0 else 0.0

            # 3. Bucle interno: Lógica "HUMANA"
            for k in range(n_data):
                val = data_matrix[k, var_idx]
                
                # CASO A: Estamos fuera del triángulo (a la izq o a la der)
                if val <= a or val >= c:
                    trainFuzzificado[k, col_idx] = 0.0
                
                # CASO B: Estamos justo en el pico
                elif val == b:
                    trainFuzzificado[k, col_idx] = 1.0
                
                # CASO C: Estamos en la subida (entre a y b)
                elif val < b:
                    # Fórmula: (x - inicio) / ancho_bajada
                    trainFuzzificado[k, col_idx] = (val - a) * inv_izq
                
                # CASO D: Estamos en la bajada (entre b y c)
                else: 
                    # Fórmula: (fin - x) / ancho_subida
                    trainFuzzificado[k, col_idx] = (c - val) * inv_der

# -------------------------------------------------------------------------
# NOMBRE: fuzzificacion_total_numba
# DESCRIPCIÓN: 
#   Procesa la base de datos completa para convertir valores reales en grados 
#   de pertenencia difusos utilizando funciones triangulares.
# ENTRADA:
#   - data_matrix [np.array]: Matriz de datos de entrenamiento.
#   - n_variables [int]: Cantidad de variables de entrada.
#   - nLabels [np.array]: Número de etiquetas por variable.
#   - x0_arr, x1_arr, x3_arr [np.array]: Matrices de vértices de los triángulos.
#   - total_labels [int]: Total de columnas de la matriz resultante.
# SALIDA: 
#   - result [np.array]: Matriz completa de pertenencia difusa (fuzzificada).
# -------------------------------------------------------------------------
@njit(fastmath=True, parallel=True)
def fuzzificacion_total_numba(data_matrix, n_variables, nLabels, x0_arr, x1_arr, x3_arr, total_labels):
    n_data = data_matrix.shape[0]
    result = np.zeros((n_data, total_labels), dtype=np.float64)
    col_idx = 0
    
    for i in range(n_variables):
        col_data = data_matrix[:, i]
        for j in range(nLabels[i]):
            p_x0 = x0_arr[i, j]
            p_x1 = x1_arr[i, j]
            p_x3 = x3_arr[i, j]
            
            inv_denom_1 = 1.0 / (p_x1 - p_x0) if (p_x1 - p_x0) != 0 else 0.0
            inv_denom_2 = 1.0 / (p_x3 - p_x1) if (p_x3 - p_x1) != 0 else 0.0
            
            for k in prange(n_data):
                val = col_data[k]
                if val <= p_x0 or val >= p_x3:
                    result[k, col_idx] = 0.0
                elif val == p_x1:
                    result[k, col_idx] = 1.0
                elif val < p_x1:
                    result[k, col_idx] = (val - p_x0) * inv_denom_1
                else: 
                    result[k, col_idx] = (p_x3 - val) * inv_denom_2
            col_idx += 1
    return result
class Fuzzy:
    # -------------------------------------------------------------------------
    # NOMBRE: set_name / set_x0 / set_x1 / set_x3 / set_y
    # DESCRIPCIÓN: 
    #   Métodos "setter" para definir las propiedades del conjunto difuso.
    # ENTRADA:
    #   - name [str], x0/x1/x3/y [float]: Valores de configuración.
    # -------------------------------------------------------------------------
    def set_name(self, name):
        self.name = name
    def set_x0(self, x0):
        self.x0 = x0
    def set_x1(self, x1):
        self.x1 = x1
    def set_x3(self, x3):
        self.x3 = x3
    def set_y(self, y):
        self.y = y

    # -------------------------------------------------------------------------
    # NOMBRE: get_x0 / get_x1 / get_x3 / get_y / getName
    # DESCRIPCIÓN: 
    #   Métodos "getter" para recuperar los parámetros del triángulo difuso.
    # SALIDA: 
    #   - [float/str]: Valor solicitado.
    # -------------------------------------------------------------------------
    def get_x0(self):
        return self.x0
    def get_x1(self):
        return self.x1
    def get_x3(self):
        return self.x3
    def get_y(self):
        return self.y
    def getName(self):
        return self.name
    
    # -------------------------------------------------------------------------
    # NOMBRE: clone
    # DESCRIPCIÓN: 
    #   Crea una copia exacta del objeto Fuzzy actual.
    # SALIDA: 
    #   - d [Fuzzy]: Nuevo objeto con los mismos parámetros.
    # -------------------------------------------------------------------------
    def clone(self):
        d = Fuzzy()
        d.x0 = self.x0
        d.x1 = self.x1
        d.x3 = self.x3
        d.y = self.y
        d.name = self.name
        return d