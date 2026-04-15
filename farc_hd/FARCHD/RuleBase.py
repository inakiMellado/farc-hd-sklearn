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

from farc_hd.FARCHD.Fuzzy import fuzzificacion_parcial
from farc_hd.FARCHD.Rule import Rule
from farc_hd.FARCHD.Itemset import Itemset
import numpy as np
from numba import njit, prange

# -------------------------------------------------------------------------
# NOMBRE: predict_bulk_wr_numba
# DESCRIPCIÓN: 
#   Motor de inferencia Winning Rule (WR) optimizado para procesar matrices completas.
#   Utiliza paralelismo (prange) para predecir miles de ejemplos simultáneamente.
# ENTRADA:
#   - X_fuzzified [np.array]: Matriz de datos fuzzificados (n_examples x n_labels).
#   - rule_antecedents [np.array]: Matriz 3D de antecedentes.
#   - clases [np.array]: Vector de consecuentes.
#   - defaultRule [int]: Clase por defecto.
#   - selected [np.array]: Máscara de reglas activas.
# SALIDA: 
#   - predictions [np.array]: Vector con las clases predichas para todos los ejemplos.
# -------------------------------------------------------------------------
@njit(fastmath=True, parallel=True)
def predict_bulk_wr_numba(X_fuzzified, rule_antecedents, clases, defaultRule, selected):
    n_examples = X_fuzzified.shape[0]
    predictions = np.full(n_examples, defaultRule, dtype=np.int32)
    n_rules = rule_antecedents.shape[0]

    # Paralelismo por ejemplo: cada hilo de CPU procesa un bloque de ejemplos
    for e in prange(n_examples): 
        max_degree = -1.0
        
        for r in range(n_rules):
            # Si hay selector y la regla está desactivada, saltar
            if selected is not None and selected[r] == 0:
                continue

            degree = 1.0
            # Cálculo del matching
            for v in range(rule_antecedents.shape[1]):
                label = rule_antecedents[r, v, 1]
                if label != -1:
                    val = X_fuzzified[e, label] # Acceso directo a la matriz batch
                    if val == 0.0:
                        degree = 0.0
                        break
                    degree *= val
            
            # Winning Rule logic: Nos quedamos con la regla que más dispare
            if degree > max_degree:
                max_degree = degree
                predictions[e] = clases[r]
                
    return predictions

# -------------------------------------------------------------------------
# NOMBRE: predict_bulk_ac_numba
# DESCRIPCIÓN: 
#   Motor de inferencia Additive Combination (AC) optimizado para matrices.
#   Suma los grados de disparo por clase para decidir el ganador.
# ENTRADA:
#   - n_classes [int]: Número total de clases del problema.
#   ... (resto igual que WR)
# -------------------------------------------------------------------------
@njit(fastmath=True, parallel=True)
def predict_bulk_ac_numba(X_fuzzified, rule_antecedents, clases, defaultRule, n_classes, selected):
    n_examples = X_fuzzified.shape[0]
    predictions = np.full(n_examples, defaultRule, dtype=np.int32)
    n_rules = rule_antecedents.shape[0]

    for e in prange(n_examples):
        degreeClass = np.zeros(n_classes, dtype=np.float64)
        
        for r in range(n_rules):
            if selected is not None and selected[r] == 0:
                continue

            degree = 1.0
            for v in range(rule_antecedents.shape[1]):
                label = rule_antecedents[r, v, 1]
                if label != -1:
                    val = X_fuzzified[e, label]
                    if val == 0.0:
                        degree = 0.0
                        break
                    degree *= val
            
            # Acumular voto para la clase de la regla
            if degree > 0.0:
                degreeClass[clases[r]] += degree

        # Decisión final para este ejemplo (Votación)
        maxDegree = -1.0
        predicted_class = defaultRule
        empate_count = 0

        for i in range(n_classes):
            if degreeClass[i] > maxDegree:
                maxDegree = degreeClass[i]
                predicted_class = i
                empate_count = 0
            elif degreeClass[i] == maxDegree and maxDegree > 0.0:
                empate_count += 1
        
        if empate_count > 0:
            predictions[e] = defaultRule
        else:
            predictions[e] = predicted_class

    return predictions

# -------------------------------------------------------------------------
# NOMBRE: compute_wracc_all_rules
# DESCRIPCIÓN: 
#   Calcula la métrica WRACC (Weighted Relative Accuracy) para todas las reglas
# ENTRADA:
#   - matchings [np.array]: Grados de activación.
#   - is_active [np.array]: Máscara de ejemplos activos.
#   - weights [np.array]: Pesos de los ejemplos.
#   - outputs [np.array]: Clases reales de los ejemplos.
#   - clases [np.array]: Clases de las reglas.
#   - frecuencias_clase [np.array]: Frecuencias de cada clase en el dataset.
# SALIDA: 
#   - wraccs [np.array]: Vector con los valores WRACC calculados.
# -------------------------------------------------------------------------
@njit(fastmath=True, parallel=True)
def compute_wracc_all_rules(matchings, is_active, weights, outputs, classes, frecuencias_clase):
    n_rules, n_examples = matchings.shape
    wraccs = np.full(n_rules, -1.0)

    for r in prange(n_rules):
        n_A = 0.0
        n_C = 0.0
        n_AC = 0.0
        target_class = classes[r]
        f_class = frecuencias_clase[r]

        for e in range(n_examples):
            if is_active[e]:
                degree = matchings[r, e]
                if degree > 0.0:
                    degree *= weights[e]
                    n_A += degree
                    if outputs[e] == target_class:
                        n_AC += degree
                        n_C += weights[e]
                elif outputs[e] == target_class:
                    n_C += weights[e]

        if n_A >= 1e-10 and n_AC >= 1e-10 and n_C >= 1e-10:
            wraccs[r] = (n_AC / n_C) * ((n_AC / n_A) - f_class)

    return wraccs

# -------------------------------------------------------------------------
# NOMBRE: compute_matchings_all_rules
# DESCRIPCIÓN: 
#   Calcula masivamente los grados de matching de todas las reglas contra 
#   todos los ejemplos del dataset.
# ENTRADA:
#   - antecedents [np.array]: Antecedentes de las reglas.
#   - dataBaseFuzzificado [np.array]: Matriz fuzzificada de la BD.
#   - confs [np.array]: Confianzas de las reglas.
# SALIDA: 
#   - matchings [np.array]: Matriz resultante de activaciones.
# -------------------------------------------------------------------------
@njit(fastmath=True, parallel=True)
def compute_matchings_all_rules(antecedents, dataBaseFuzzificado, confs):
    n_rules = antecedents.shape[0]
    n_examples = dataBaseFuzzificado.shape[0]
    matchings = np.zeros((n_rules, n_examples))

    for r in prange(n_rules):
        conf = confs[r]
        for e in range(n_examples):
            degree = 1.0
            for v in range(antecedents.shape[1]):
                var_label = antecedents[r, v, 1]
                if var_label >= 0:
                    degree *= dataBaseFuzzificado[e, var_label]
                    if degree == 0.0:
                        break
            matchings[r, e] = degree * conf
    return matchings

# -------------------------------------------------------------------------
# NOMBRE: reduce_weight_numba
# DESCRIPCIÓN: 
#   Actualiza el estado de activación de los ejemplos basándose en la 
#   cobertura de las reglas seleccionadas.
# ENTRADA:
#   - is_active [np.array]: Estado actual de los ejemplos.
#   - counts [np.array]: Contador de veces que cada ejemplo ha sido cubierto.
#   - outputs [np.array]: Clases de los ejemplos.
#   - matching [np.array]: Activación de la mejor regla.
#   - clas [int]: Clase de la regla.
#   - K [int]: Umbral de cobertura.
# SALIDA: 
#   - is_active, counts, count_removed: Estado actualizado y contador de eliminados.
# -------------------------------------------------------------------------
@njit(fastmath=True)
def reduce_weight_numba(is_active, counts, outputs, matching, clas, K):
    count_removed = 0
    # recorremos todas las reglas
    for i in range(len(is_active)):
        # Si la regla sigue activa
        if is_active[i]:
            # si el matching calculado es > 0.0
            if matching[i] > 0.0:
                counts[i] += 1
                # Si la regla ha sido cubierta >= k y clase de la regla == clase
                if counts[i] >= K and outputs[i] == clas:
                    is_active[i] = False
                    count_removed += 1
    return is_active, counts, count_removed

# -------------------------------------------------------------------------
# NOMBRE: obtenerLabelsUsadas_numba
# DESCRIPCIÓN: 
#   Identifica qué etiquetas difusas están siendo utilizadas por las reglas 
#   activas para optimizar procesos posteriores.
# ENTRADA:
#   - selected [np.array]: Reglas activas.
#   - rule_antecedents [np.array]: Antecedentes globales.
#   - n_labels_total [int]: Total de etiquetas en la BD.
# SALIDA: 
#   - reglasIncidir [np.array]: Matriz de etiquetas en uso.
# -------------------------------------------------------------------------
@njit(fastmath=True)
def obtenerLabelsUsadas_numba(selected, rule_antecedents, n_labels_total):
    reglasIncidir = np.full((n_labels_total, 3), -1, dtype=np.int32)
    n_rules = len(selected)
    for i in range(n_rules):
        if selected[i] == 1:
            for v in range(rule_antecedents.shape[1]):
                local_val = rule_antecedents[i, v, 0]
                global_val = rule_antecedents[i, v, 1]
                if global_val != -1:
                    reglasIncidir[global_val, 0] = v           
                    reglasIncidir[global_val, 1] = local_val   
                    reglasIncidir[global_val, 2] = global_val  
    return reglasIncidir

# -------------------------------------------------------------------------
# NOMBRE: evaluate_core_numba
# DESCRIPCIÓN: 
#   Motor de evaluación masiva. Calcula la asociación de todas las reglas 
#   activas con todos los ejemplos y retorna la precisión global del sistema.
# ENTRADA:
#   - matchings [np.array]: Grados de activación precalculados.
#   - pesos [np.array]: Pesos (confianza) de las reglas.
#   - selected [np.array]: Reglas activas en el cromosoma.
#   - clases [np.array]: Clases de las reglas.
#   - clases_unicas [np.array]: Lista de clases distintas.
#   - clases_ejemplos [np.array]: Clases reales del dataset.
#   - typeInference [int]: 0 para Máximo (WR), 1 para Suma (AC).
#   - defaultRule [int]: Clase por defecto.
#   - asociacion, asClases, predicciones: Buffers de memoria.
# SALIDA: 
#   - fitness [float]: Precisión del sistema (0-100).
#   - nUncover [int]: Número de ejemplos no cubiertos por ninguna regla.
# -------------------------------------------------------------------------
@njit(fastmath=True)
def evaluate_core_numba(matchings, pesos, selected, clases, clases_unicas, clases_ejemplos, typeInference, defaultRule, asociacion, asClases, predicciones):
    n_rules, n_examples = matchings.shape
    n_clases = len(clases_unicas)

    # Iterar todas las reglas
    for i in range(n_rules):
        # Si esta activo en el cromosoma
        if selected[i] == 1:
            w = pesos[i]
            # Iterar todos los ejemplos
            for e in range(n_examples):
                # Aplicar el peso de la regla al grado de matching
                asociacion[i, e] = matchings[i, e] * w
        # Si no lo esta, no cuenta
        else:
            for e in range(n_examples):
                asociacion[i, e] = 0.0

    asClases[:] = 0.0
    # Iterar por cada regla
    for r in range(n_rules):
        # Si esta activo en el cromosoma
        if selected[r] == 1:
            c = clases[r]
            c_idx = -1
            # Buscar clase de la regla
            for k in range(n_clases):
                if clases_unicas[k] == c:
                    c_idx = k
                    break
            # Si la clase tiene regla, aplicar inferencia 
            if c_idx != -1:
                # valor MÁXIMO
                if typeInference == 0: 
                    for e in range(n_examples):
                        val = asociacion[r, e]
                        if val > asClases[c_idx, e]:
                            asClases[c_idx, e] = val
                # suma de valores
                else: 
                     for e in range(n_examples):
                        asClases[c_idx, e] += asociacion[r, e]

    # Iniciar con una regla por defecto
    predicciones[:] = defaultRule
    nUncover = 0
    # por cada ejemplo
    for e in prange(n_examples):
        max_val = -1.0
        max_c_idx = -1
        empate = False
        all_zeros = True
        # Clase con mayor puntuación
        for k in range(n_clases):
            val = asClases[k, e]
            if val > 0.0:
                all_zeros = False
            
            # valor con mejor puntuacion
            if val > max_val:
                max_val = val
                max_c_idx = k
                empate = False
            # resolver empate
            elif val == max_val and val > 0.0:
                empate = True
        # valor no cubierto
        if all_zeros:
            nUncover += 1
        # Si no hay empate y hay un ganador, predecimos la clase
        # De haber empate, seria defaultRule
        elif not empate and max_c_idx != -1:
            predicciones[e] = clases_unicas[max_c_idx]
    # calular fitness        
    hits = 0
    for e in prange(n_examples):
        if predicciones[e] == clases_ejemplos[e]:
            hits += 1
            
    return (hits / n_examples) * 100.0, nUncover

# -------------------------------------------------------------------------
# NOMBRE: calc_compatibilidad_indices
# DESCRIPCIÓN: 
#   Calcula el grado de compatibilidad de todas las reglas seleccionadas 
#   contra el dataset fuzzificado de forma paralela.
# ENTRADA:
#   - rule_antecedents [np.array]: Antecedentes.
#   - selected [np.array]: Reglas activas.
#   - trainFuzzificado [np.array]: BD fuzzificada.
#   - compatibilidad [np.array]: Buffer de salida.
# SALIDA: 
#   - compatibilidad [np.array]: Matriz de compatibilidad actualizada.
# -------------------------------------------------------------------------
@njit(fastmath=True, parallel=True)
def calc_compatibilidad_indices(rule_antecedents, selected, trainFuzzificado, compatibilidad):
    n_rules = rule_antecedents.shape[0]
    n_examples = trainFuzzificado.shape[0]
    n_vars = rule_antecedents.shape[1]

    # Paralelismo por reglas
    for regla in prange(n_rules):
        # Si la regla no está seleccionada, saltamos
        if selected[regla] == 0:
            continue
        # Flag para optimizar la inicialización
        initialized = False

        for variable in range(n_vars):
            label = rule_antecedents[regla, variable, 1]

            # Si la variable participa en la regla
            if label != -1:
                if not initialized:
                    # Primera variable
                    for e in range(n_examples):
                        compatibilidad[regla, e] = trainFuzzificado[e, label]
                    initialized = True
                else:
                    # Siguientes variables
                    for e in range(n_examples):
                        compatibilidad[regla, e] *= trainFuzzificado[e, label]
        
        # Si la regla no tenía antecedentes (caso raro, regla vacía), es 1.0
        if not initialized:
            for e in range(n_examples):
                compatibilidad[regla, e] = 1.0
                
    return compatibilidad

class RuleBase:
# -------------------------------------------------------------------------
    # NOMBRE: __init__
    # DESCRIPCIÓN: 
    #   Constructor de la clase RuleBase. Configura el motor de inferencia y 
    #   establece las estructuras de datos necesarias para la evaluación masiva.
    # ENTRADA:
    #   - dataBase [DataBase]: base de datos difusa 
    #   - train [myDataSet]: Conjunto de datos de entrenamiento.
    #   - K [int]: Factor de recubrimiento. 
    #   - typeInference [int]: Selector del Método de Razonamiento Difuso (FRM).
    #   - pesos [np.array]: Vector de confianzas precalculadas
    # -------------------------------------------------------------------------
    def __init__(self, dataBase=None, train=None, K=None, typeInference=None, pesos = None):
        self.ruleBase = []
        self.dataBase = dataBase
        self.train = train
        self.n_variables = dataBase.numVariables() if dataBase else 0
        self.fitness = 0.0
        self.K = K
        self.typeInference = typeInference
        self.defaultRule = -1
        self.nUncover = 0
        self.nUncoverClas = [0] * (train.getnClasses() if train else 0)
        self.clases = np.array([])
        self.pesos = np.array([])
        
        # Obtiene antecedentes y los prepara para que numba los pueda usar mas rapido
        # Asi no trabajamos con objetos de tipo rule
        self.cached_antecedents = None
        
        # Antes se calculaban cada vez, era una pequeña ineficiencia
        if train is not None:
            self.clases_unicas = np.unique(train.getOutputsAsIntegers())
        else:
            self.clases_unicas = np.array([])

        # Evitan pedir todo el rato memoria al sistema (evitamos gestrion de memoria = menos tiempo ejecucion)
        self.buffer_compatibilidad = None
        self.buffer_asociacion = None
        self.buffer_asClases = None
        self.buffer_predicciones = None

    # -------------------------------------------------------------------------
    # NOMBRE: clone
    # DESCRIPCIÓN: 
    #   Crea una copia superficial de la base de reglas manteniendo las 
    #   referencias a los datos pesados pero con su propia lista de reglas.
    # SALIDA: 
    #   - rb [RuleBase]: base de reglas.
    # -------------------------------------------------------------------------
    def clone(self):
        rb = RuleBase(self.dataBase, self.train, self.K, self.typeInference, self.pesos)
        rb.ruleBase = self.ruleBase[:]
        rb.nUncover = self.nUncover
        rb.nUncoverClas = self.nUncoverClas[:]
        rb.clases = self.clases[:]
        if self.cached_antecedents is not None:
            rb.cached_antecedents = self.cached_antecedents 
        rb.clases_unicas = self.clases_unicas 
        return rb

    # -------------------------------------------------------------------------
    # NOMBRE: add
    # DESCRIPCIÓN: 
    #   Añade una regla, otra RuleBase o un Itemset a la base actual.
    # ENTRADA:
    #   - insertar [Rule|RuleBase|Itemset]: Elemento a añadir.
    # -------------------------------------------------------------------------
    def add(self, insertar):
        if isinstance(insertar, Rule):
            self.ruleBase.append(insertar)
        elif isinstance(insertar, RuleBase):
            self.ruleBase.extend(insertar.ruleBase)
        elif isinstance(insertar, Itemset):
            antecedent = np.full((self.n_variables, 2), -1, dtype=np.int32)
            for i in range(insertar.size()):
                item = insertar.get(i)
                antecedent[item.getVariable()] = [item.getValue(), item.getVariableValueAbs()]

            r = Rule(self.dataBase)
            r.asignaAntecedente(antecedent)
            r.setConsequent(insertar.getClas())
            r.setConfidence(insertar.getSupportClass() / insertar.getSupport() if insertar.getSupport() > 0 else 0)
            r.setSupport(insertar.getSupportClass())
            self.ruleBase.append(r)

        self.clases = np.array([regla.getClas() for regla in self.ruleBase], dtype=np.int32)
        self.cached_antecedents = None
        # Reseteamos buffers porque el número de reglas ha cambiado
        self.resetear_comp_asoc_buffers()

    # -------------------------------------------------------------------------
    # NOMBRE: get / size / sort / remove / clear
    # DESCRIPCIÓN: 
    #   Métodos estándar para la manipulación y consulta de la lista de reglas.
    # ------------------------------------------------------------------------
    def get(self, pos):
        return self.ruleBase[pos]

    def size(self):
        return len(self.ruleBase)

    def sort(self):
        self.ruleBase.sort()
        self.cached_antecedents = None
    
    def remove(self, pos):
        if isinstance(self.ruleBase, list):
            self.ruleBase.pop(pos)
        else:
             self.ruleBase = np.delete(self.ruleBase, pos, axis=0)
             
        self.clases = np.delete(self.clases, pos)
        self.cached_antecedents = None
        self.resetear_comp_asoc_buffers()

    def clear(self):
        self.ruleBase = []
        self.fitness = 0.0
        self.clases = np.array([])
        self.cached_antecedents = None
        self.resetear_comp_asoc_buffers()

    # -------------------------------------------------------------------------
    # NOMBRE: resetear_comp_asoc_buffers
    # DESCRIPCIÓN: 
    #   Invalida los buffers de compatibilidad y asociación para forzar 
    #   su recalculación.
    # -------------------------------------------------------------------------
    def resetear_comp_asoc_buffers(self):
        # Solo buffers que dependan de n_reglas (compatibilidad y asociacion)
        self.buffer_compatibilidad = None
        self.buffer_asociacion = None
    
    # -------------------------------------------------------------------------
    # NOMBRE: getTypeInference / getAccuracy / getK / getUncover / hasUncover
    # DESCRIPCIÓN: 
    #   Getters para diversos parámetros y estados de la base de reglas.
    # -------------------------------------------------------------------------
    def getTypeInference(self):
        return self.typeInference
    def getAccuracy(self):
        return self.fitness
    def getK(self):
        return self.K
    def getUncover(self):
        return self.nUncover
    def hasUncover(self):
        return self.nUncover > 0
    
    # -------------------------------------------------------------------------
    # NOMBRE: setDefaultRule
    # DESCRIPCIÓN: 
    #   Establece la regla por defecto basándose en la clase mayoritaria 
    #   del conjunto de entrenamiento.
    # -------------------------------------------------------------------------
    def setDefaultRule(self):
        bestRule = 0
        for i in range(self.train.getnClasses()):
            if self.train.numberInstances(bestRule) < self.train.numberInstances(i):
                bestRule = i
        self.defaultRule = bestRule

    # -------------------------------------------------------------------------
    # NOMBRE: sync_numba_buffers
    # DESCRIPCIÓN: 
    #   Transforma la lista de objetos Rule (Python) en matrices densas de NumPy. 
    #   Esta sincronización es crítica para que Numba pueda acceder a los datos 
    #   en memoria contigua y ejecutar los cálculos a velocidad de lenguaje nativo.
    # -------------------------------------------------------------------------
    def sync_numba_buffers(self):
        # Sincroniza la lista de objetos Rule con los arrays de NumPy para Numba
        # Solo actua si antecedentes es nulo (no se han guardado) o si su tamaño cambio respecto al numero de reglas
        # Antes se hacia via mascara de reglas
        if self.cached_antecedents is None or len(self.cached_antecedents) != len(self.ruleBase):
            if len(self.ruleBase) == 0:
                self.cached_antecedents = np.zeros((0, self.n_variables, 2), dtype=np.int32)
                self.clases = np.array([], dtype=np.int32)
                self.pesos = np.array([], dtype=np.float64)
            else:
                # RECOPILA los antecedentes de todos los objetos Rule en un solo Array 3D
                self.cached_antecedents = np.array([r.antecedent for r in self.ruleBase], dtype=np.int32)
                # EXTRAE las clases de todos los objetos en un Array 1D
                self.clases = np.array([r.clas for r in self.ruleBase], dtype=np.int32)
                if len(self.pesos) != len(self.ruleBase):
                    self.almacenaPesos()

    # -------------------------------------------------------------------------
    # NOMBRE: iniciar_buffers
    # DESCRIPCIÓN: 
    #   Gestiona la memoria estática asignando espacio a los buffers de trabajo. 
    #   Reutiliza los arrays si el tamaño no ha cambiado, evitando la sobrecarga 
    #   de reservar memoria en cada iteración del algoritmo genético.
    # ENTRADA:
    #   - n_rules [int]: Número actual de reglas en la base.
    #   - n_examples [int]: Número de ejemplos en el dataset de entrenamiento.
    #   - n_clases [int]: Cantidad de clases distintas en el problema.
    # -------------------------------------------------------------------------
    def iniciar_buffers(self, n_rules, n_examples, n_clases):
        # Como trabajamos con buffers, solo se inician cuando estos no hayan sido usados

        # Verificar buffer_compatibilidad (Rules x Examples)
        if self.buffer_compatibilidad is None or self.buffer_compatibilidad.shape != (n_rules, n_examples):
            self.buffer_compatibilidad = np.zeros((n_rules, n_examples), dtype=np.float64)
            self.buffer_asociacion = np.zeros((n_rules, n_examples), dtype=np.float64)
        
        # Verificar buffer_asClases (Classes x Examples)
        if self.buffer_asClases is None or self.buffer_asClases.shape != (n_clases, n_examples):
            self.buffer_asClases = np.zeros((n_clases, n_examples), dtype=np.float64)

        # Verificar buffer_predicciones (Examples)
        if self.buffer_predicciones is None or self.buffer_predicciones.shape[0] != n_examples:
            self.buffer_predicciones = np.zeros(n_examples, dtype=np.int32)

    # -------------------------------------------------------------------------
    # NOMBRE: evaluate
    # DESCRIPCIÓN: 
    #   Realiza la evaluación masiva de la base de reglas. Si se proporciona un 
    #   cromosoma (gene), sintoniza la base de datos antes de calcular el fitness.
    # ENTRADA:
    #   - gene [np.array]: Cromosoma del algoritmo genético con los desplazamientos laterales.
    #   - selected [np.array]: Vector binario que indica qué reglas están activas.
    # -------------------------------------------------------------------------
    def evaluate(self, gene=None, selected=None):
        if len(self.ruleBase) == 0:
            self.fitness = 0.0
            self.nUncover = self.train.getnData()
            return

        self.sync_numba_buffers()

        if selected is None:
            selected = np.ones(len(self.ruleBase), dtype=np.int32)
        else:
            selected = selected.astype(np.int32)

        if gene is not None:
            self.dataBase.decode(gene)
            mascaraLabelsUsadas = self.obtenerLabelsUsadas(selected)
            self.prefuzzyGA(mascaraLabelsUsadas)
        
        clases_ejemplos = self.dataBase.getClasses()
        
        n_rules = len(self.ruleBase)
        n_examples = len(clases_ejemplos)
        n_clases = len(self.clases_unicas)
        
        self.iniciar_buffers(n_rules, n_examples, n_clases)
        # Se calculan compatibilidades de cada regla
        calc_compatibilidad_indices(
            self.cached_antecedents, 
            selected, 
            self.dataBase.trainFuzzificado, 
            self.buffer_compatibilidad
        )
        # se calculan el fitness y reglas no cubiertas
        self.fitness, self.nUncover = evaluate_core_numba(
            self.buffer_compatibilidad, 
            self.pesos, 
            selected, 
            self.clases, 
            self.clases_unicas, 
            clases_ejemplos, 
            self.typeInference, 
            self.defaultRule,
            self.buffer_asociacion,
            self.buffer_asClases,
            self.buffer_predicciones
        )
   
    # -------------------------------------------------------------------------
    # NOMBRE: obtenerLabelsUsadas
    # DESCRIPCIÓN: 
    #   Filtra y devuelve las etiquetas lingüísticas que realmente están 
    #   presentes en las reglas activas para optimizar la fuzzificación.
    # ENTRADA:
    #   - selected [np.array]: Máscara de reglas seleccionadas.
    # SALIDA: 
    #   - [np.array]: Matriz con los índices de etiquetas que deben procesarse.
    # -------------------------------------------------------------------------
    def obtenerLabelsUsadas(self, selected):
        if len(self.ruleBase) == 0:
            return np.full((self.dataBase.labelTotales, 3), -1, dtype=np.int32)
        return obtenerLabelsUsadas_numba(selected, self.cached_antecedents, self.dataBase.labelTotales)
    
    # -------------------------------------------------------------------------
    # NOMBRE: prefuzzyGA
    # DESCRIPCIÓN: 
    #   Realiza una fuzzificación parcial para los cambios realizados por el algoritmo genético.
    # ENTRADA:
    #   - valores [np.array]: Índices de las etiquetas que necesitan ser recalculadas.
    # -------------------------------------------------------------------------
    def prefuzzyGA(self, valores):
        x0_arr = self.dataBase.x0_arr
        x1_arr = self.dataBase.x1_arr
        x3_arr = self.dataBase.x3_arr
        
        data_matrix = self.train.getAllData()
        trainFuzzificado = self.dataBase.trainFuzzificado
        
        fuzzificacion_parcial(valores, data_matrix, x0_arr, x1_arr, x3_arr, trainFuzzificado)

    # -------------------------------------------------------------------------
    # NOMBRE: reduceRules
    # DESCRIPCIÓN: 
    #   Aplica un algoritmo de selección voraz para reducir el tamaño de la 
    #   base de reglas. Selecciona iterativamente las reglas con mejor WRACC 
    #   (Weighted Relative Accuracy) hasta cumplir el criterio de cobertura K 
    #   para la clase objetivo.
    # ENTRADA:
    #   - clase_objetivo [int]: El índice de la clase
    # -------------------------------------------------------------------------
    def reduceRules(self, clase_objetivo):
        """
        Reduce la base de reglas seleccionando iterativamente las mejores reglas (WRACC)
        hasta cubrir los ejemplos de la clase objetivo o agotar las reglas.
        """
        # print(f"Entra en reducir reglas para la clase: {clase_objetivo} con reglas: {len(self.ruleBase)}")
        if len(self.ruleBase) == 0:
            return
        # Eliminamos la sobrecarga de objetos para permitir operaciones vectorizadas.
        num_total_datos = self.train.getnData()
        
        # 'veces_cubierto': Cuántas veces ha sido cubierto cada ejemplo
        veces_cubierto = np.zeros(num_total_datos, dtype=np.int32)
        
        # 'pesos_ejemplos': Peso actual de cada ejemplo (inicialmente 1.0)
        pesos_ejemplos = np.ones(num_total_datos, dtype=np.float64) 
        
        # 'ejemplos_activos_mask': Booleano, True si el ejemplo aún debe ser considerado
        ejemplos_activos_mask = np.ones(num_total_datos, dtype=np.bool_)
        
        # 'reglas_seleccionadas': Booleano, marca qué reglas ya hemos escogido
        reglas_seleccionadas = np.zeros(len(self.ruleBase), dtype=np.bool_)
        
        # 'ejemplos_por_cubrir': Contador de ejemplos que faltan por cubrir de la clase objetivo
        ejemplos_por_cubrir = self.train.numberInstances(clase_objetivo)
        num_reglas_seleccionadas = 0
        
        # Sincronizamos buffers de memoria para Numba (necesario para funciones numba)
        self.sync_numba_buffers()
        
        # Calculamos TODOS los grados de emparejamiento una sola vez
        todas_confianzas = np.array([r.conf for r in self.ruleBase], dtype=np.float64)
        
        todos_matchings = compute_matchings_all_rules(
            self.cached_antecedents, 
            self.dataBase.trainFuzzificado, 
            todas_confianzas
        )

        # Datos para realizar los calculos (clases del problema, frecuencia de las clases, clase real de cada ejemplo)
        todas_clases_reglas = self.clases
        todas_frecuencias_clase = np.array([self.train.frecuentClass(c) for c in todas_clases_reglas], dtype=np.float64)
        etiquetas_reales = self.train.getOutputsAsIntegers()

        while True:
            # 1. Identificar reglas candidatas (no seleccionadas)-> reglas_seleccionadas == False
            indices_reglas_no_seleccionadas = np.where(reglas_seleccionadas == False)[0]
            
            if len(indices_reglas_no_seleccionadas) == 0:
                break # No quedan reglas disponibles

            # 2. Crear mascaras booleanas para las reglas candidatas
            matchings_reglas_no_seleccionadas = todos_matchings[indices_reglas_no_seleccionadas]
            clases_candidatas = todas_clases_reglas[indices_reglas_no_seleccionadas]
            frecuencias_candidatas = todas_frecuencias_clase[indices_reglas_no_seleccionadas]
            
            # 3. Calcular métrica WRACC masivamente usando Numba
            todos_wraccs = compute_wracc_all_rules(
                matchings_reglas_no_seleccionadas, 
                ejemplos_activos_mask, 
                pesos_ejemplos, 
                etiquetas_reales, 
                clases_candidatas, 
                frecuencias_candidatas
            )

            # 4. Seleccionar la mejor regla de este ciclo
            idx_relativo_mejor = np.argmax(todos_wraccs)
            mejor_valor_wracc = todos_wraccs[idx_relativo_mejor]

            if mejor_valor_wracc <= -1.0: # Umbral de parada (ninguna regla aporta valor)
                break

            # 5. Marcar la regla ganadora como seleccionada y +1 en cantidad de reglas seleccionadas
            idx_global_mejor = indices_reglas_no_seleccionadas[idx_relativo_mejor]
            reglas_seleccionadas[idx_global_mejor] = True
            num_reglas_seleccionadas += 1

            # 6. Actualizar contadores y desactivar ejemplos cubiertos
            matching_mejor_regla = todos_matchings[idx_global_mejor]
            clase_mejor_regla = todas_clases_reglas[idx_global_mejor]
            
            ejemplos_activos_mask, veces_cubierto, num_ejemplos_eliminados = reduce_weight_numba(
                ejemplos_activos_mask,
                veces_cubierto,
                etiquetas_reales, 
                matching_mejor_regla, 
                clase_mejor_regla, 
                self.K
            )

            # 7. Actualización vectorizada de pesos
            mascara_limite_k = veces_cubierto >= self.K
            pesos_ejemplos = 1.0 / (veces_cubierto + 1.0)
            pesos_ejemplos[mascara_limite_k] = 0.0
            
            # 8. Comprobamos si se cubrieron suficientes ejemplos
            ejemplos_por_cubrir -= num_ejemplos_eliminados
            if ejemplos_por_cubrir <= 0 or num_reglas_seleccionadas >= len(self.ruleBase):
                break
        
        total_instancias_clase = self.train.numberInstances(clase_objetivo)
        ejemplos_cubiertos = total_instancias_clase - ejemplos_por_cubrir
        # print(f"Sale de reduce Reglas para la clase {clase_objetivo}: "
              # f"Numero examples: {ejemplos_cubiertos}/{total_instancias_clase}. "
              # f"Numero de reglas: {num_reglas_seleccionadas}/{len(self.ruleBase)}. "
              # f"Valor de K: {self.K}")
        # print()
        
        # Al salir del bucle filtramos reglas y clases 
        self.ruleBase = [self.ruleBase[i] for i in range(len(self.ruleBase)) if reglas_seleccionadas[i]]
        self.clases = np.array([r.clas for r in self.ruleBase], dtype=np.int32)
        self.almacenaPesos()
        
        # Limpieza de caché (necesaria para numba)
        self.cached_antecedents = None
        self.resetear_comp_asoc_buffers()
    # -------------------------------------------------------------------------
    # NOMBRE: almacenaPesos
    # DESCRIPCIÓN: 
    #   Sincroniza los valores de confianza de los objetos Rule con el array 
    #   de pesos numéricos utilizado por los kernels de evaluación.
    # -------------------------------------------------------------------------
    def almacenaPesos(self):
        self.pesos = np.array([self.ruleBase[i].getConfidence() for i in range(len(self.ruleBase))])
    
    # -------------------------------------------------------------------------
    # NOMBRE: predict_dataset
    # DESCRIPCIÓN: 
    #   **NUEVO MÉTODO** para Scikit-Learn.
    #   Reemplaza al antiguo FRM(). Recibe una matriz fuzzificada completa y 
    #   devuelve todas las predicciones usando los kernels vectorizados (Bulk).
    # ENTRADA:
    #   - X_fuzzified [np.array]: Datos de entrada fuzzificados.
    #   - selected [np.array]: (Opcional) Máscara de reglas a usar.
    # SALIDA: 
    #   - [np.array]: Clases predichas.
    # -------------------------------------------------------------------------
    def predict_dataset(self, X_fuzzified, selected=None):
        # 1. Asegurar buffers de numba
        self.sync_numba_buffers()
        
        # 2. Casteo de selector si es necesario
        selected_arr = None
        if selected is not None:
            selected_arr = selected.astype(np.int32)
        
        # 3. Obtener n_classes (necesario para AC)
        if self.train:
             n_classes = self.train.getnClasses()
        else:
             # Fallback si no hay train (ej: modelo cargado de disco)
             n_classes = len(self.clases_unicas)

        # 4. Llamada al Kernel correspondiente
        if self.typeInference == 0: # Winning Rule
            return predict_bulk_wr_numba(
                X_fuzzified, 
                self.cached_antecedents, 
                self.clases, 
                self.defaultRule, 
                selected_arr
            )
        else: # Additive Combination
            return predict_bulk_ac_numba(
                X_fuzzified, 
                self.cached_antecedents, 
                self.clases, 
                self.defaultRule, 
                n_classes,
                selected_arr
            )

    # -------------------------------------------------------------------------
    # NOMBRE: printString
    # DESCRIPCIÓN: 
    #   Genera una representación textual legible de toda la base de reglas actual,
    #   incluyendo el conteo total y el listado enumerado de cada regla.
    # SALIDA: 
    #   - stringOut [str]: Cadena lista para ser impresa o guardada en un archivo de texto.
    # -------------------------------------------------------------------------
    def printString(self):
        stringOut = f"@Number of rules: {len(self.ruleBase)}\n"
        for i, rule in enumerate(self.ruleBase):
            stringOut += f"{i+1}: {rule}\n"
        return stringOut
    
    # -------------------------------------------------------------------------
    # NOMBRE: setOriginalNamesToRules
    # DESCRIPCIÓN: 
    #   Asigna nombres reales creados por el usuario a reglas.
    # -------------------------------------------------------------------------
    def setOriginalNamesToRules (self, variables, classes):
        # Valores reales de numero de variables, clases y etiquetas
        n_vars = self.n_variables
        n_classes = self.dataBase.getnClases() 
        n_labels_per_var = self.dataBase.getnLabels()
        
        # Variables
        valid_vars = (variables is not None) and (len(variables) == n_vars)
        if not valid_vars and variables is not None:
            print(f"Error numero de Variables: Esperados {n_vars}, Recibidos {len(variables)}")

        # Clases
        valid_classes = (classes is not None) and (len(classes) == n_classes)
        if not valid_classes and classes is not None:
             print(f"Error numero de Clases: Esperados {n_classes}, Recibidos {len(classes)}")

        if valid_vars or valid_classes:
            for rule in self.ruleBase:
                if valid_vars:
                    rule.setRealVariableNames(variables)
                if valid_classes:
                    rule.setRealClassNames(classes)


    # -------------------------------------------------------------------------
    # NOMBRE: saveFile
    # DESCRIPCIÓN: 
    #   Exporta la base de reglas completa a un archivo de texto.
    # ENTRADA:
    #   - filename [str]: Ruta y nombre del archivo de destino.
    # -------------------------------------------------------------------------
    def saveFile(self, filename):
        with open(filename, 'w') as f:
            f.write(self.printString())