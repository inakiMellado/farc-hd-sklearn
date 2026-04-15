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

from farc_hd.FARCHD.Item import Item
import numpy as np
from numba import njit, prange

# -------------------------------------------------------------------------
# NOMBRE: calc_support_numba
# DESCRIPCIÓN: 
#   Calcula el soporte total y el soporte de clase de un itemset.
# ENTRADA:
#   - trainFuzzy [np.array]: Matriz global de grados de pertenencia.
#   - indices [np.array]: Índices absolutos de las etiquetas del itemset.
#   - classes [np.array]: Vector con las etiquetas de clase reales.
#   - target_class [int]: Clase objetivo para la cual calculamos el soporte.
# SALIDA: 
#   - sum_total [float]: Suma acumulada de grados de pertenencia (soporte).
#   - sum_class [float]: Suma acumulada solo para los ejemplos de la clase objetivo.
# -------------------------------------------------------------------------
@njit(fastmath=True, parallel=True)
def calc_support_numba(trainFuzzy, indices, classes, target_class):
    n_examples = trainFuzzy.shape[0]
    n_items = len(indices)
    
    sum_total = 0.0
    sum_class = 0.0
    
    # Paralelismo por ejemplos
    for i in prange(n_examples):
        idx0 = indices[0]
        degree = trainFuzzy[i, idx0]
        # Solo iteramos si el degree inicial es > 0 y hay más items
        # La condición degree > 0.0 actúa como poda (sparsity)
        if n_items > 1 and degree > 0.0:
            for j in range(1, n_items):
                idx = indices[j]
                val = trainFuzzy[i, idx]
                degree *= val
                # Si el grado cae a 0, paramos inmediatamente
                if degree == 0.0:
                    break
        
        sum_total += degree
        if classes[i] == target_class:
            sum_class += degree
            
    return sum_total, sum_class

class Itemset:
    # -------------------------------------------------------------------------
    # NOMBRE: __init__
    # DESCRIPCIÓN: 
    #   Inicializa una lista vacía de ítems
    # ENTRADA:
    #   - clas [int]: Clase a la que se asocia este conjunto de ítems.
    # -------------------------------------------------------------------------
    def __init__ (self, clas = None):
        self.itemset = [] 
        self.clas = clas
        self.support = 0.0
        self.supportRule = 0.0
        
        # Cache para evitar reconstruir el array de índices para Numba
        self.cached_indices = None
    
    # -------------------------------------------------------------------------
    # NOMBRE: clone
    # DESCRIPCIÓN: 
    #   Realiza una copia del itemset
    # SALIDA: 
    #   - d [Itemset]: Nuevo objeto clonado.
    # -------------------------------------------------------------------------
    def clone(self):
        d = Itemset(self.clas)
        # Clonamos la lista y los objetos Item dentro
        d.itemset = [item.clone() for item in self.itemset]
        d.clas = self.clas
        d.support = self.support
        d.supportRule = self.supportRule
        # No clonamos cached_indices, se regenerará si hace falta
        return d

    # -------------------------------------------------------------------------
    # NOMBRE: add / remove / get / size
    # DESCRIPCIÓN: 
    #   Gestión de la lista de ítems. Invalidan la caché interna al modificar
    #   la estructura del conjunto.
    # -------------------------------------------------------------------------
    def add (self, item):
        self.itemset.append(item)
        # Invalidamos la caché porque los items han cambiado
        self.cached_indices = None
    
    def get (self, pos):
        return self.itemset[pos]
    
    def remove (self, pos):
        # Eliminamos y invalidamos caché
        del self.itemset[pos]
        self.cached_indices = None
        return self.itemset 

    def size (self):
        return len(self.itemset)
    
    def getSupport(self):
        return self.support
    
    def getSupportClass(self):
        return self.supportRule
    
    def getClas(self):
        return self.clas

    def setClas(self, clas):
        self.clas = clas
    
    def __eq__(self, other):
        if len(self.itemset) != len(other.itemset) or self.clas != other.clas:
            return False
        # Comparación eficiente de listas
        for i in range(len(self.itemset)):
            if self.itemset[i] != other.itemset[i]:
                return False
        return True

    # -------------------------------------------------------------------------
    # NOMBRE: calculateSupports
    # DESCRIPCIÓN: 
    #   Traduce los objetos Item a arrays Numpy y llama al kernel de Numba.
    # ENTRADA:
    #   - dataBase [DataBase]: Base de datos para obtener los valores fuzzificados.
    # -------------------------------------------------------------------------
    def calculateSupports(self, dataBase):
        # Si no hay items, no hay soporte (evitar errores)
        if len(self.itemset) == 0:
            self.support = 0.0
            self.supportRule = 0.0
            return
        # Solo construimos el array numpy si no existe en caché.
        if self.cached_indices is None:
            self.cached_indices = np.array([item.variableValueAbs for item in self.itemset], dtype=np.int32)
        
        # Obtenemos los datos necesarios de la base de datos
        trainFuzzificado = dataBase.getTrainFuzzy()
        clasesSalida = dataBase.getClasses()
        
        # Llamada al kernel optimizado de Numba
        totalComp, sumaClase = calc_support_numba(
            trainFuzzificado, 
            self.cached_indices, 
            clasesSalida, 
            self.clas
        )

        n_examples = len(clasesSalida)
        if n_examples > 0:
            self.support = totalComp / n_examples
            self.supportRule = sumaClase / n_examples
        else:
            self.support = 0.0
            self.supportRule = 0.0

    # -------------------------------------------------------------------------
    # NOMBRE: degree
    # DESCRIPCIÓN: 
    #   LLama a degreeProduct
    # ENTRADA:
    #   - dataBase: DB para consulta de matching.
    #   - examplePos [int]: Índice del ejemplo en el dataset.
    # -------------------------------------------------------------------------
    def degree(self, dataBase, ejemploPos):
        return self.degreeProduct(dataBase, ejemploPos)

    # -------------------------------------------------------------------------
    # NOMBRE: degreeProduct
    # DESCRIPCIÓN: 
    #   Calcula el grado de activación (T-norma producto) del itemset para 
    #   un ejemplo concreto.
    # ENTRADA:
    #   - dataBase: DB para consulta de matching.
    #   - examplePos [int]: Índice del ejemplo en el dataset.
    # SALIDA: 
    #   - degree [float]: Valor de activación en el rango [0, 1].
    # -------------------------------------------------------------------------
    def degreeProduct(self, dataBase, examplePos):
        degree = 1.0
        # Usamos caché si está disponible para acceso rápido a variableValueAbs
        if self.cached_indices is not None:
            for idx in self.cached_indices:
                val = dataBase.matching(idx, examplePos)
                degree *= val
                if degree <= 0.0:
                    return 0.0
        else:
            for item in self.itemset:
                val = dataBase.matching(item.variableValueAbs, examplePos)
                degree *= val
                if degree <= 0.0:
                    return 0.0
        return degree