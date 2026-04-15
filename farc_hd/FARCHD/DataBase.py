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
from farc_hd.FARCHD.Fuzzy import Fuzzy, fuzzificacion_total_numba
from farc_hd.org.core.Files import Files
from numba import njit


# -------------------------------------------------------------------------
# NOMBRE: decode_numba
# DESCRIPCIÓN: 
#   Realiza el desplazamiento de las funciones de pertenencia triangulares 
#   basándose en un cromosoma (gen).
# ENTRADA:
#   - gene [np.array]: Vector de valores que dictan el desplazamiento.
#   - n_variables [int]: Número total de variables de entrada.
#   - nLabels [np.array]: Cantidad de etiquetas por variable.
#   - varReal [np.array]: Máscara booleana para identificar variables continuas.
#   - ini_x0, ini_x1, ini_x3: Matrices con puntos base del triángulo.
#   - output_x0, output_x1, output_x3: Matrices donde se guardan los puntos modificados.
# -------------------------------------------------------------------------
@njit(fastmath=True)
def decode_numba(gene, n_variables, nLabels, varReal, ini_x0, ini_x1, ini_x3, output_x0, output_x1, output_x3):
    pos = 0
    for i in range(n_variables):
        # Omitimos variables que no son reales/continuas
        if not varReal[i]:
            continue
            
        count_labels = nLabels[i]
        for j in range(count_labels):
            val_gene = gene[pos]
            nuevaPos = val_gene - 0.5
            x1_curr = ini_x1[i, j]
            
            # CASO 1: Es la primera etiqueta (solo tiene vecinos a la derecha)
            if j == 0:
                x1_next = ini_x1[i, j+1]
                displacement = nuevaPos * (x1_next - x1_curr)
            
            # CASO 2: Es la última etiqueta (solo tiene vecinos a la izquierda)
            elif j == (count_labels - 1):
                x1_prev = ini_x1[i, j-1]
                displacement = nuevaPos * (x1_curr - x1_prev)
            
            # CASO 3: Etiquetas intermedias
            else:
                x1_next = ini_x1[i, j+1]
                x1_prev = ini_x1[i, j-1]
                # Si se mueve a la derecha, el rango máximo es hacia el centro del vecino derecho
                if val_gene >= 0.5:
                    displacement = nuevaPos * (x1_next - x1_curr)
                # Si se mueve a la izquierda, el rango máximo es hacia el centro del vecino izquierdo
                else:
                    displacement = nuevaPos * (x1_curr - x1_prev)
            
            # Aplicamos el desplazamiento calculado a los tres puntos del triángulo (A, B, C)
            output_x0[i, j] = ini_x0[i, j] + displacement
            output_x1[i, j] = ini_x1[i, j] + displacement
            output_x3[i, j] = ini_x3[i, j] + displacement
            
            # Avanzamos a la siguiente posición del cromosoma
            pos += 1

class DataBase:
    # -------------------------------------------------------------------------
    # NOMBRE: __init__
    # DESCRIPCIÓN: 
    #   Crea la partición difusa inicial (grid uniforme) y prepara los arrays
    #   Numpy para el procesamiento acelerado.
    # ENTRADA:
    #   - nLabels [int]: Número por defecto de etiquetas difusas por variable.
    #   - train [myDataSet]: Conjunto de entrenamiento para calcular rangos.
    # -------------------------------------------------------------------------
    def __init__(self, nLabels, train):
        self.todasLasClases = train.getOutputsAsIntegers()
        self.nClases = train.getnClasses()
        self.nEtiquetas = nLabels
        ranks = train.returnRanks()
        self.n_variables = train.getnInputs()
        self.names = np.copy(train.names())
        self.nLabels = np.zeros(self.n_variables, dtype=np.int32)
        self.varReal = np.array([False]*self.n_variables)
        self.dataBase = [None] * self.n_variables
        self.dataBaseIni = [None] * self.n_variables
        self.labelTotales = 0
        
        # Definición de Etiquetas y Estructura Fuzzy
        for i in range (self.n_variables):
            rank = np.abs(ranks[i][1] - ranks[i][0])
            
            # Determinamos el número de etiquetas según el tipo de variable
            if train.isNominal(i):
                self.nLabels[i] = int(rank)+1
            elif (train.isInteger(i)) and ((rank+1) <= nLabels):
                self.nLabels[i] = int(rank)+1
            else:
                # Si es continua, usamos el parámetro nLabels y marcamos como sintonizable (varReal)
                self.nLabels[i] = nLabels
                self.varReal[i] = True
            
            self.dataBase[i] = [None]*self.nLabels[i]
            self.dataBaseIni[i] = [None]*self.nLabels[i]
            mark = rank / (self.nLabels[i] - 1.0)
            
            for j in range(self.nLabels[i]):
                self.dataBase[i][j] = Fuzzy()
                self.dataBaseIni[i][j] = Fuzzy()

                # Definición de los 3 puntos del triángulo (x0:izq, x1:centro, x3:der)
                value = ranks[i][0] + (mark * (j-1))
                x0_val = self.setValue(value, ranks[i][0], ranks[i][1])
                
                value = ranks[i][0] + (mark * j)
                x1_val = self.setValue(value, ranks[i][0], ranks[i][1])
                
                value = ranks[i][0] + (mark * (j+1))    
                x3_val = self.setValue(value, ranks[i][0], ranks[i][1])

                # Guardamos en la base de datos actual e inicial
                self.dataBaseIni[i][j].set_x0(x0_val)
                self.dataBaseIni[i][j].set_x1(x1_val)
                self.dataBaseIni[i][j].set_x3(x3_val)
                self.dataBaseIni[i][j].set_y(1.0)
                
                self.dataBase[i][j].set_x0(x0_val)
                self.dataBase[i][j].set_x1(x1_val)
                self.dataBase[i][j].set_x3(x3_val)
                self.dataBase[i][j].set_y(1.0)

                nombre = "L_" + str(j) + "(" + str(self.nLabels[i]) + ")"
                self.dataBase[i][j].set_name(nombre)
                self.dataBaseIni[i][j].set_name(nombre)

                self.labelTotales += 1

        
        # Preparación de Matrices para Optimización (Numba)
        # Creamos matrices densas para evitar el acceso lento a listas de objetos Fuzzy
        max_labels = np.max(self.nLabels)
        
        # Matrices para valores iniciales (constantes durante la evolución)
        self.ini_x0 = np.zeros((self.n_variables, max_labels), dtype=np.float64)
        self.ini_x1 = np.zeros((self.n_variables, max_labels), dtype=np.float64)
        self.ini_x3 = np.zeros((self.n_variables, max_labels), dtype=np.float64)
        
        # Matrices actualizables (se actualizan en cada iteración de decode_numba)
        self.x0_arr = np.zeros((self.n_variables, max_labels), dtype=np.float64)
        self.x1_arr = np.zeros((self.n_variables, max_labels), dtype=np.float64)
        self.x3_arr = np.zeros((self.n_variables, max_labels), dtype=np.float64)
        
        # Volcamos los datos de los objetos Fuzzy a las matrices de NumPy
        for i in range(self.n_variables):
            for j in range(self.nLabels[i]):
                self.ini_x0[i, j] = self.dataBaseIni[i][j].get_x0()
                self.ini_x1[i, j] = self.dataBaseIni[i][j].get_x1()
                self.ini_x3[i, j] = self.dataBaseIni[i][j].get_x3()
                
                self.x0_arr[i, j] = self.dataBase[i][j].get_x0()
                self.x1_arr[i, j] = self.dataBase[i][j].get_x1()
                self.x3_arr[i, j] = self.dataBase[i][j].get_x3()

    # -------------------------------------------------------------------------
    # NOMBRE: getnClases
    # DESCRIPCIÓN: 
    #   Retorna el numero de clases
    # SALIDA: 
    #   - self.nClases[int]: numero de clases del dataset.
    # -------------------------------------------------------------------------
    def getnClases(self):
        return self.nClases

    # -------------------------------------------------------------------------
    # NOMBRE: setValue
    # DESCRIPCIÓN: 
    #   Ajusta valores con una tolerancia de 1E-4 para evitar errores de 
    #   precisión numérica en los límites de los rangos.
    # ENTRADA:
    #   - val, minimo, tope [float]: Valor a evaluar y límites.
    # SALIDA: 
    #   - val [float]: Valor ajustado o el original.
    # -------------------------------------------------------------------------
    def setValue(self, val, minimo, tope):
        if (val > (minimo - 1E-4)) and (val < (minimo + 1E-4)):
            return minimo
        if (val > (tope - 1E-4)) and (val < (tope + 1E-4)):
            return tope
        return val
    
    # -------------------------------------------------------------------------
    # NOMBRE: decode
    # DESCRIPCIÓN: 
    #   Actualiza las matrices numéricas de los triángulos difusos 
    # ENTRADA:
    #   - gene [np.array]: Genotipo del individuo a decodificar.
    # -------------------------------------------------------------------------
    def decode(self, gene):
        # Solo actualizamos las matrices de números (x0_arr, x1_arr, etc.)
        # Esto es lo que usa Numba para evaluar rápido.
        decode_numba(gene, self.n_variables, self.nLabels, self.varReal, 
                     self.ini_x0, self.ini_x1, self.ini_x3,
                     self.x0_arr, self.x1_arr, self.x3_arr)

    # -------------------------------------------------------------------------
    # NOMBRE: synchronize_objects
    # DESCRIPCIÓN: 
    #   Sincroniza el "mundo Numpy" con el "mundo de objetos Python". Actualiza
    #   cada objeto Fuzzy individual con los valores calculados en los arrays.
    # -------------------------------------------------------------------------
    def synchronize_objects(self):
        # Copia los valores de los arrays optimizados de vuelta a los objetos Fuzzy
        # para que funciones como 'printString' o 'saveFile' funcionen bien.
        for i in range(self.n_variables):
            if not self.varReal[i]:
                continue
            for j in range(self.nLabels[i]):
                # Pasamos los datos del mundo Numpy al mundo de Objetos Python
                self.dataBase[i][j].set_x0(self.x0_arr[i, j])
                self.dataBase[i][j].set_x1(self.x1_arr[i, j])
                self.dataBase[i][j].set_x3(self.x3_arr[i, j])

    # -------------------------------------------------------------------------
    # NOMBRE: numVariables
    # DESCRIPCIÓN: 
    #   Retorna el numero de variables
    # SALIDA: 
    #   - self.n_variables[int]: numero de variables.
    # -------------------------------------------------------------------------
    def numVariables(self):
        return self.n_variables
    
    # -------------------------------------------------------------------------
    # NOMBRE: getnLabelsReal
    # DESCRIPCIÓN: 
    #   Calcula el número total de etiquetas difusas para variables continuas.
    # SALIDA: 
    #   - count [int]: Suma de etiquetas de variables reales.
    # -------------------------------------------------------------------------
    def getnLabelsReal(self):
        count = 0
        for i in range (self.n_variables):
            if self.varReal[i]:
                count += self.nLabels[i]
        return count

    # -------------------------------------------------------------------------
    # NOMBRE: numLabels
    # DESCRIPCIÓN: 
    #   Retorna la cantidad de etiquetas de una variable específica.
    # ENTRADA:
    #   - variable [int]: Índice de la variable.
    # SALIDA: 
    #   - [int]: Número de etiquetas.
    # -------------------------------------------------------------------------
    def numLabels(self, variable):
        return self.nLabels[variable]

    # -------------------------------------------------------------------------
    # NOMBRE: getnLabels
    # DESCRIPCIÓN: 
    #   Retorna el array con la cantidad de etiquetas de todas las variables.
    # SALIDA: 
    #   - self.nLabels[np.array]: Vector de enteros con las etiquetas por variable.
    # -------------------------------------------------------------------------
    def getnLabels(self):
        return self.nLabels

    # -------------------------------------------------------------------------
    # NOMBRE: matching
    # DESCRIPCIÓN: 
    #   Retorna el grado de pertenencia precalculado para un ejemplo y etiqueta.
    # ENTRADA:
    #   - variableLabel [int]: Índice de la etiqueta en la matriz de fuzzificación.
    #   - value [int]: Índice del ejemplo.
    # SALIDA: 
    #   - [float]: Valor de matching entre 0 y 1.
    # -------------------------------------------------------------------------
    def matching(self, variableLabel, value):
        if (variableLabel < 0):
            return 1
        else:
            return self.trainFuzzificado[value][variableLabel]
    
    # -------------------------------------------------------------------------
    # NOMBRE: print_triangle
    # DESCRIPCIÓN: 
    #   Genera una cadena de texto con los parámetros del triángulo de una etiqueta.
    # ENTRADA:
    #   - var [int]: Índice de la variable.
    #   - label [int]: Índice de la etiqueta.
    # SALIDA: 
    #   - cadena [str]: Información formateada del triángulo.
    # -------------------------------------------------------------------------
    def print_triangle(self, var, label):
        cadena = ""
        d = self.dataBase[var][label]
        cadena += d.getName() + ": \t" + str(d.get_x0()) + "\t" + str(d.get_x1()) + "\t" + str(d.get_x3()) + "\n"
        return cadena

    # -------------------------------------------------------------------------
    # NOMBRE: print
    # DESCRIPCIÓN: 
    #   Retorna el nombre asignado a una etiqueta específica.
    # ENTRADA:
    #   - var [int]: Índice de la variable.
    #   - label [int]: Índice de la etiqueta.
    # SALIDA: 
    #   - [str]: Nombre de la etiqueta.
    # -------------------------------------------------------------------------
    def print(self, var, label):
        return self.dataBase[var][label].getName()

    # -------------------------------------------------------------------------
    # NOMBRE: printString
    # DESCRIPCIÓN: 
    #   Genera un resumen completo de la base de datos difusa en formato texto.
    # SALIDA: 
    #   - string [str]: Resumen detallado de variables y etiquetas.
    # -------------------------------------------------------------------------
    def printString(self):
        string = "@Using Triangular Membership Functions as antecedent fuzzy sets"
        for i in range (self.n_variables):
            string += "\n\n@Number of Labels in Variable " + str((i+1)) + ": " + str(self.nLabels[i])
            string += "\n" + self.names[i] + ":\n"
            for j in range(self.nLabels[i]):
                string += self.dataBase[i][j].getName() + ": (" + str(self.dataBase[i][j].get_x0()) + ", " + str(self.dataBase[i][j].get_x1()) + ", " + str(self.dataBase[i][j].get_x3()) + ")\n"
        return string

    # -------------------------------------------------------------------------
    # NOMBRE: saveFile
    # DESCRIPCIÓN: 
    #   Guarda la configuración actual de la base de datos en un archivo físico.
    # ENTRADA:
    #   - filename [str]: Ruta del archivo de destino.
    # -------------------------------------------------------------------------
    def saveFile(self, filename):
        stringOut = self.printString()
        Files.writeFile(filename, stringOut)
    
    # -------------------------------------------------------------------------
    # NOMBRE: fuzzificacion
    # DESCRIPCIÓN: 
    #   Calcula los grados de pertenencia de todos los datos de 
    #   entrada frente a las etiquetas de la base de datos.
    # ENTRADA:
    #   - datos [myDataSet]: El conjunto de datos a procesar.
    # -------------------------------------------------------------------------
    def fuzzificacion(self, datos):
        datos_raw = datos.getAllData()
        self.trainFuzzificado = fuzzificacion_total_numba(
            datos_raw,
            self.n_variables,
            self.nLabels,
            self.x0_arr,
            self.x1_arr,
            self.x3_arr,
            self.labelTotales
        )
    
    # -------------------------------------------------------------------------
    # NOMBRE: getTrainFuzzy
    # DESCRIPCIÓN: 
    #   Retorna la matriz de datos ya fuzzificados.
    # SALIDA: 
    #   - [np.array]: Matriz de grados de pertenencia.
    # -------------------------------------------------------------------------
    def getTrainFuzzy(self) -> np.array:
        return self.trainFuzzificado
        
    # -------------------------------------------------------------------------
    # NOMBRE: getClasses
    # DESCRIPCIÓN: 
    #   Retorna el vector con las clases reales de cada ejemplo del dataset.
    # SALIDA: 
    #   - [np.array]: Vector de enteros con las clases.
    # -------------------------------------------------------------------------
    def getClasses(self) -> np.array:
        return self.todasLasClases