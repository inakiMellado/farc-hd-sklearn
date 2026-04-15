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


class Item:
    # -------------------------------------------------------------------------
    # NOMBRE: __init__
    # DESCRIPCIÓN: 
    #   Eelaciona entre una variable y su etiqueta difusa correspondiente.
    # ENTRADA:
    #   - variable [int]: Índice de la variable de entrada.
    #   - value [int]: Índice de la etiqueta difusa.
    #   - absV [int]: Valor absoluto único asignado al par variable-valor.
    # SALIDA: 
    #   - None
    # -------------------------------------------------------------------------
    def __init__(self, variable = None, value = None, absV = None):
        if variable is not None and value is not None and absV is not None:
            self.variable = variable
            self.value = value
            self.variableValueAbs = absV
    
    # -------------------------------------------------------------------------
    # NOMBRE: setValues
    # DESCRIPCIÓN: 
    #   Permite actualizar de forma conjunta los atributos del ítem.
    # ENTRADA:
    #   - variable [int]: Nuevo índice de variable.
    #   - value [int]: Nuevo índice de valor difuso.
    #   - absV [int]: Nuevo valor absoluto.
    # SALIDA: 
    #   - None
    # -------------------------------------------------------------------------
    def setValues (self, variable, value, absV):
        self.variable = variable
        self.value = value
        self.variableValueAbs = absV
    
    # -------------------------------------------------------------------------
    # NOMBRE: getVariable / getValue / getVariableValueAbs
    # DESCRIPCIÓN: 
    #   Métodos de acceso (getters) para recuperar las propiedades del ítem.
    # SALIDA: 
    #   - [int]: El atributo solicitado.
    # -------------------------------------------------------------------------
    def getVariable (self):
        return self.variable

    def getValue (self):
        return self.value

    def getVariableValueAbs (self):
        return self.variableValueAbs

    # -------------------------------------------------------------------------
    # NOMBRE: clone
    # DESCRIPCIÓN: 
    #   Crea una copia del ítem actual.
    # SALIDA: 
    #   - d [Item]: El nuevo objeto clonado.
    # -------------------------------------------------------------------------
    def clone(self):
        d = Item()
        d.variable = self.variable
        d.value = self.value
        d.variableValueAbs = self.variableValueAbs
        return d
  
    # -------------------------------------------------------------------------
    # NOMBRE: __lt__ (Less Than)
    # DESCRIPCIÓN: 
    #   Define el criterio de ordenación para los ítems. Prioriza el índice de 
    #   la variable para mantener un orden coherente en los itemsets.
    # ENTRADA:
    #   - other [Item]: Ítem con el que se compara.
    # SALIDA: 
    #   - [bool]: Resultado de la comparación.
    # -------------------------------------------------------------------------
    def __lt__(self, other):
        # Primero comparar por 'variable'
        if other.variable > self.variable:
            return True
        elif other.variable < self.variable:
            return False
        elif other.value > self.value:
            return False
        elif other.value < self.value:
            return True

    # -------------------------------------------------------------------------
    # NOMBRE: __eq__ (Equality)
    # DESCRIPCIÓN: 
    #   Determina si dos ítems son iguales basándose en la coincidencia de 
    #   variable y valor difuso.
    # ENTRADA:
    #   - other [Item]: Ítem con el que se compara.
    # SALIDA: 
    #   - [bool]: True si representan la misma relación variable-valor.
    # -------------------------------------------------------------------------
    def __eq__(self, other):
        return self.variable == other.variable and self.value == other.value
  

