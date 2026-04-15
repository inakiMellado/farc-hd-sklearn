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
from farc_hd.org.core.Randomize import Randomize
from numba import njit

# -------------------------------------------------------------------------
# NOMBRE: string_rep_numba
# DESCRIPCIÓN: 
#   Calcula una medida de diferencia basada en la representación de bits 
#   (Gray/Binary) de los genes reales. Se utiliza para el cálculo de 
#   distancias entre individuos.
# ENTRADA:
#   - gene1, gene2 [np.array]: Cromosomas reales a comparar.
#   - BITS_GEN [int]: Número de bits utilizados para codificar cada gen.
#   - nGenes [int]: Cantidad total de genes en el cromosoma.
# SALIDA: 
#   - count [int]: Número de bits diferentes encontrados.
# -------------------------------------------------------------------------
@njit(fastmath=True)
def string_rep_numba(gene1, gene2, BITS_GEN, nGenes):
    count = 0
    INCREMENTO = 1.0 / ((1 << BITS_GEN) - 1)
    for idx in range(nGenes):
        val1 = int(gene1[idx] / INCREMENTO + 0.5)
        val2 = int(gene2[idx] / INCREMENTO + 0.5)
        prev_bit1 = -1
        prev_bit2 = -1
        for b in range(BITS_GEN - 1, -1, -1):
            bit1 = (val1 >> b) & 1
            bit2 = (val2 >> b) & 1
            if prev_bit1 != -1:
                diff_bit1 = bit1 ^ prev_bit1
                diff_bit2 = bit2 ^ prev_bit2
            else:
                diff_bit1 = bit1
                diff_bit2 = bit2
            if diff_bit1 != diff_bit2:
                count += 1
            prev_bit1 = bit1
            prev_bit2 = bit2
    return count

# -------------------------------------------------------------------------
# NOMBRE: xPC_BLX_numba
# DESCRIPCIÓN: 
#   Implementa el cruce BLX-alpha (Blend Crossover) para genes reales.
# ENTRADA:
#   - gene_self/gene_indiv [np.array]: Cromosomas de los padres.
#   - d [float]: Parámetro alpha que define la amplitud del cruce.
#   - nGenes [int]: Número de genes reales.
#   - generator: Generador de números aleatorios JIT.
# SALIDA: 
#   - None (Modifica los genes in-place)
# -------------------------------------------------------------------------
@njit(fastmath=True)
def xPC_BLX_numba(gene_self, gene_indiv, d, nGenes, generator):
    # generator es la instancia de MTwister
    for i in range(nGenes):      
        diff = abs(gene_self[i] - gene_indiv[i])
        I = d * diff
        
        A1 = gene_self[i] - I
        if A1 < 0.0:
            A1 = 0.0
        
        C1 = gene_self[i] + I
        if C1 > 1.0:
            C1 = 1.0
        
        # LLAMADA AL GENERADOR ORIGINAL COMPILADO
        rnd = generator.genrand_res53() 
        gene_self[i] = A1 + rnd * (C1 - A1)
        
        A2 = gene_indiv[i] - I
        if A2 < 0.0: A2 = 0.0
        C2 = gene_indiv[i] + I
        if C2 > 1.0: C2 = 1.0
        
        rnd = generator.genrand_res53()
        gene_indiv[i] = A2 + rnd * (C2 - A2)

# -------------------------------------------------------------------------
# NOMBRE: two_point_crossover_numba
# DESCRIPCIÓN: 
#   Operador de cruce de dos puntos. Intercambia el segmento de genes 
#   comprendido entre dos posiciones elegidas al azar.
# ENTRADA:
#   - gene_self/gene_indiv [np.array]: Cromosomas de los padres.
#   - nGenes [int]: Longitud del cromosoma.
#   - generator: Instancia del generador aleatorio.
# -------------------------------------------------------------------------
@njit(fastmath=True)
def two_point_crossover_numba(gene_self, gene_indiv, nGenes, generator):
    # Replicamos Randint logic: low + (high-low)*rnd
    # low=0, high=nGenes
    r1 = generator.genrand_res53()
    p1 = int(0 + (nGenes - 0) * r1)
    
    r2 = generator.genrand_res53()
    p2 = int(0 + (nGenes - 0) * r2)

    if p2 < p1:
        p1, p2 = p2, p1
                        
    for i in range(p1, p2 + 1):                 
        temp = gene_self[i]
        gene_self[i] = gene_indiv[i]
        gene_indiv[i] = temp

# -------------------------------------------------------------------------
# NOMBRE: hux_numba
# DESCRIPCIÓN: 
#   Cruce HUX (Half Uniform Crossover) para la parte binaria (selección de reglas). 
#   Intercambia exactamente la mitad de los bits que son diferentes entre padres.
# ENTRADA:
#   - geneR_self/geneR_indiv [np.array]: Cromosomas binarios de selección.
#   - generator: Generador de números aleatorios JIT.
# -------------------------------------------------------------------------
@njit(fastmath=True)
def hux_numba(geneR_self, geneR_indiv, generator):
    n = len(geneR_self)
    diff_indices = []
    for i in range(n):
        if geneR_self[i] != geneR_indiv[i]:
            diff_indices.append(i)
    
    dist = len(diff_indices)
    nPos = dist // 2
    
    if nPos > 0:
        indices_arr = np.array(diff_indices)
        for i in range(nPos):
            rnd = generator.genrand_res53()
            idx_rand = int(dist * rnd)
            
            # Simulación exacta de la lógica HUX original:
            pos_real = indices_arr[idx_rand]
            
            # Intercambio de genes
            temp_val = geneR_self[pos_real]
            geneR_self[pos_real] = geneR_indiv[pos_real]
            geneR_indiv[pos_real] = temp_val
            
            # Decremento dist y swap en indices_arr para simular "sacar" el indice
            dist -= 1
            temp_idx = indices_arr[dist]
            indices_arr[dist] = indices_arr[idx_rand]
            indices_arr[idx_rand] = temp_idx

# -------------------------------------------------------------------------
# NOMBRE: random_values_gene_numba
# DESCRIPCIÓN: 
#   Genera valores aleatorios uniformes en el rango [0, 1] para un 
#   cromosoma real.
# ENTRADA:
#   - gene_arr [np.array]: Array a rellenar con valores aleatorios.
#   - generator: Instancia del generador aleatorio compilado.
# -------------------------------------------------------------------------
@njit(fastmath=True)
def random_values_gene_numba(gene_arr, generator):
    for i in range(len(gene_arr)):
        gene_arr[i] = generator.genrand_res53()

# -------------------------------------------------------------------------
# NOMBRE: random_values_geneR_numba
# DESCRIPCIÓN: 
#   Genera valores aleatorios binarios (0 o 1) con una probabilidad del 
#   50% para la selección de reglas.
# ENTRADA:
#   - geneR_arr [np.array]: Array a rellenar con ceros y unos.
#   - generator: Instancia del generador aleatorio compilado.
# SALIDA: 
#   - None
# -------------------------------------------------------------------------
@njit(fastmath=True)
def random_values_geneR_numba(geneR_arr, generator):
    for i in range(len(geneR_arr)):
        if generator.genrand_res53() < 0.5:
            geneR_arr[i] = 0
        else:
            geneR_arr[i] = 1

# -------------------------------------------------------------------------
# NOMBRE: distHamming_geneR_numba
# DESCRIPCIÓN: 
#   Calcula distancia hamming entre 2 cromosomas
# ENTRADA:
#   - arr1/arr2 [np.array]: Cromosomas a calcular.
# SALIDA: 
#   - count[int]: veces que pos de arr1 != pos arr2
# -------------------------------------------------------------------------
@njit(fastmath=True)
def distHamming_geneR_numba(arr1, arr2):
    count = 0
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            count += 1
    return count

# -------------------------
# CLASE INDIVIDUAL
# -------------------------

class Individual:
    
    # -------------------------------------------------------------------------
    # NOMBRE: __init__
    # DESCRIPCIÓN: 
    #   Inicializa un individuo con sus dos partes genéticas: 'gene' (real para 
    #   sintonización lateral) y 'geneR' (binario para selección de reglas).
    # ENTRADA:
    #   - ruleBase: Base de reglas asociada.
    #   - dataBase: Base de datos difusa.
    #   - w1 [float]: Peso de importancia (fitness).
    # -------------------------------------------------------------------------
    def __init__(self, ruleBase = None, dataBase = None, w1 = None):
        if ruleBase is None and dataBase is None and w1 is None:
            return
        self.ruleBase = ruleBase
        self.w1 = w1
        self.fitness = -np.inf
        self.accuracy = 0.0
        self.n_e = 0
        self.nGenes = dataBase.getnLabelsReal()
        
        if self.nGenes > 0:
            self.gene = np.zeros(self.nGenes, dtype=np.float64)
        else:
            self.gene = None
        self.geneR = np.zeros(self.ruleBase.size(), dtype=np.int32)

    # -------------------------------------------------------------------------
    # NOMBRE: clone
    # DESCRIPCIÓN: 
    #   Crea una copia del individuo actual
    # SALIDA: 
    #   - ind [Individual]: Nuevo objeto idéntico al actual con sus propios arrays.
    # -------------------------------------------------------------------------
    def clone(self):
        ind = Individual()
        ind.ruleBase = self.ruleBase
        ind.w1 = self.w1
        ind.fitness = self.fitness
        ind.accuracy = self.accuracy
        ind.n_e = self.n_e
        ind.nGenes = self.nGenes
        if self.nGenes > 0:
            ind.gene = self.gene.copy()
        else:
            ind.gene = None
        ind.geneR = self.geneR.copy()
        return ind
    
    # -------------------------------------------------------------------------
    # NOMBRE: reset
    # DESCRIPCIÓN: 
    #   Reinicia los cromosomas a sus valores neutros
    # -------------------------------------------------------------------------
    def reset(self):
        if self.nGenes > 0:
            self.gene += 0.5
        self.geneR += 1

    # -------------------------------------------------------------------------
    # NOMBRE: randomValues
    # DESCRIPCIÓN: 
    #   Asigna valores aleatorios iniciales a ambos cromosomas del individuo.
    # -------------------------------------------------------------------------
    def randomValues(self):
        gen = Randomize.getGenerator()
        if self.nGenes > 0:
            random_values_gene_numba(self.gene, gen)
        random_values_geneR_numba(self.geneR, gen)

    # -------------------------------------------------------------------------
    # NOMBRE: size
    # DESCRIPCIÓN: 
    #   Retorna el tamaño del cromosoma binario (cantidad total de reglas).
    # SALIDA: 
    #   - [int]: Longitud de self.geneR.
    # -------------------------------------------------------------------------
    def size(self):
        return len(self.geneR)

    # -------------------------------------------------------------------------
    # NOMBRE: getnSelected
    # DESCRIPCIÓN: 
    #   Cuenta cuántas reglas han sido seleccionadas (activadas) en el cromosoma.
    # SALIDA: 
    #   - [int]: Número de genes en geneR con valor mayor a 0.
    # -------------------------------------------------------------------------
    def getnSelected(self):
        return np.count_nonzero(self.geneR > 0)

    # -------------------------------------------------------------------------
    # NOMBRE: isNew / onNew / offNew
    # DESCRIPCIÓN: 
    #   Gestión del flag n_e que indica si un individuo ha sido modificado
    #   y necesita ser re-evaluado.
    # -------------------------------------------------------------------------
    def isNew (self):
        return self.n_e == 1
    def onNew (self):
        self.n_e = 1
    def offNew (self):
        self.n_e = 0

    # -------------------------------------------------------------------------
    # NOMBRE: setw1 / getAccuracy / getFitness
    # DESCRIPCIÓN: 
    #   Métodos de acceso (getters/setters) para propiedades de rendimiento.
    # -------------------------------------------------------------------------
    def setw1 (self, value):
        self.w1 = value
    def getAccuracy(self):
        return self.accuracy
    def getFitness(self):
        return self.fitness

    # -------------------------------------------------------------------------
    # NOMBRE: StringRep
    # DESCRIPCIÓN: 
    #   Calcula la representación en cadena de bits para medir diversidad.
    # ENTRADA:
    #   - indiv [Individual]: Individuo con el que comparar.
    #   - BITS_GEN [int]: Precisión de la codificación real.
    # SALIDA: 
    #   - [int]: Número de bits de diferencia.
    # -------------------------------------------------------------------------
    def StringRep(self, indiv, BITS_GEN):
        return string_rep_numba(self.gene, indiv.gene, BITS_GEN, self.nGenes)
    
    # -------------------------------------------------------------------------
    # NOMBRE: distHamming
    # DESCRIPCIÓN: 
    #   Calcula la distancia de Hamming 
    # ENTRADA:
    #   - ind [Individual]: El otro individuo de la comparación.
    #   - BITS_GEN [int]: Precisión binaria para la parte real.
    # SALIDA: 
    #   - count [int]: Distancia total calculada.
    # -------------------------------------------------------------------------
    def distHamming(self, ind, BITS_GEN):
        count = distHamming_geneR_numba(self.geneR, ind.geneR)
        if self.nGenes > 0:
            count += string_rep_numba(self.gene, ind.gene, BITS_GEN, self.nGenes)
        return count

    # -------------------------------------------------------------------------
    # NOMBRE: Hux / xPC_BLX / twoPoint
    # DESCRIPCIÓN: 
    #   Llamadas a los operadores de cruce optimizados con Numba.
    # ENTRADA:
    #   - indiv [Individual]: Individuo pareja para el cruce.
    #   - d [float]: Parámetro para BLX (solo en xPC_BLX).
    # -------------------------------------------------------------------------
    def Hux(self, indiv):
        hux_numba(self.geneR, indiv.geneR, Randomize.getGenerator())
    def xPC_BLX(self, indiv, d):
        if self.nGenes > 0:
            xPC_BLX_numba(self.gene, indiv.gene, d, self.nGenes, Randomize.getGenerator())
    def twoPoint(self, indiv):
        if self.nGenes > 0:
            two_point_crossover_numba(self.gene, indiv.gene, self.nGenes, Randomize.getGenerator())

    # -------------------------------------------------------------------------
    # NOMBRE: generateRB
    # DESCRIPCIÓN: 
    #   Decodifica los cromosomas para generar una Base de Reglas física,
    #   eliminando las reglas inactivas según el cromosoma geneR.
    # SALIDA: 
    #   - ruleBase [RuleBase]: Base de reglas sintonizada y podada.
    # -------------------------------------------------------------------------
    def generateRB(self):
        ruleBase = self.ruleBase.clone()
        ruleBase.almacenaPesos()
        ruleBase.evaluate(self.gene, self.geneR)
        ruleBase.setDefaultRule()
        
        # Eliminar reglas con geneR < 1
        # Usamos iteración inversa para borrar de la lista
        for i in range (len(self.geneR)-1, -1, -1):
            if self.geneR[i] < 1:
                ruleBase.remove(i)
        return ruleBase
    
    # -------------------------------------------------------------------------
    # NOMBRE: evaluate
    # DESCRIPCIÓN: 
    #   Invoca la evaluación de la base de reglas con la configuración
    #   genética actual y actualiza la precisión y el fitness.
    # -------------------------------------------------------------------------
    def evaluate(self):
        self.ruleBase.evaluate(self.gene, self.geneR)
        self.accuracy = self.ruleBase.getAccuracy()
        self.fitness = self.accuracy
    
    # -------------------------------------------------------------------------
    # NOMBRE: __lt__ / __eq__
    # DESCRIPCIÓN: 
    #   Operadores de comparación basados en el fitness para permitir 
    #   la ordenación automática de la población.
    # -------------------------------------------------------------------------
    def __lt__(self, other):
        if other.fitness < self.fitness:
            return True
        elif other.fitness > self.fitness:
            return False
        return False
    def __eq__(self, other):
        return self.fitness == other.fitness