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

from farc_hd.FARCHD.DataBase import DataBase
import numpy as np

class Rule:
    # -------------------------------------------------------------------------
    # NOMBRE: __init__
    # DESCRIPCIÓN: 
    #   Puede inicializar una regla vacía a partir de una DataBase o crear
    #   una copia de otra regla existente.
    # ENTRADA:
    #   - insertar [Rule | DataBase]: Objeto para inicializar o clonar la regla.
    # SALIDA: 
    #   - None
    # -------------------------------------------------------------------------
    def __init__(self, insertar):
        if isinstance(insertar, Rule):
            self.antecedent = insertar.antecedent.copy()
            self.clas = insertar.clas
            self.dataBase = insertar.dataBase
            self.conf = insertar.conf
            self.supp = insertar.supp
            self.nAnts = insertar.nAnts
            self.wracc = insertar.wracc
        elif isinstance(insertar, DataBase):
            self.antecedent = np.zeros((insertar.numVariables(), 2), dtype=np.int32) -1
            self.clas = -1
            self.dataBase = insertar
            self.conf = 0.0
            self.supp = 0.0
            self.nAnts = 0
            self.wracc = 0.0
    
    # -------------------------------------------------------------------------
    # NOMBRE: clone
    # DESCRIPCIÓN: 
    #   Crea una réplica exacta de la regla actual.
    # SALIDA: 
    #   - r [Rule]: Nueva instancia de la regla clonada.
    # -------------------------------------------------------------------------
    def clone (self):
        r = Rule(self.dataBase)
        r.antecedent = self.antecedent.copy()
        r.clas = self.clas
        r.conf = self.conf
        r.supp = self.supp
        r.nAnts = self.nAnts
        r.wracc = self.wracc
        return r
    
    # -------------------------------------------------------------------------
    # NOMBRE: asignaAntecedente
    # DESCRIPCIÓN: 
    #   Copia un antecedente externo a la regla y actualiza el contador de 
    #   proposiciones activas (nAnts).
    # ENTRADA:
    #   - antecedent [np.array]: Matriz de antecedentes.
    # -------------------------------------------------------------------------
    def asignaAntecedente(self, antecedent):
        self.nAnts = 0
        for i in range(antecedent.shape[0]):
            self.antecedent[i] = antecedent[i]
            if (self.antecedent[i,1] > -1):
                self.nAnts+=1

    # -------------------------------------------------------------------------
    # NOMBRE: matching
    # DESCRIPCIÓN: 
    #   Llama al degreeProduct
    # ENTRADA:
    #   - examplePos [int]: Índice del ejemplo.
    # -------------------------------------------------------------------------
    def matching(self, examplePos):
        return self.degreeProduct(examplePos)

    # -------------------------------------------------------------------------
    # NOMBRE: degreeProduct
    # DESCRIPCIÓN: 
    #   Calcula el grado de activación de la regla para un ejemplo mediante 
    #   la T-norma del producto, ponderada por la confianza.
    # ENTRADA:
    #   - examplePos [int]: Índice del ejemplo en el dataset.
    # SALIDA: 
    #   - [float]: Grado de verdad resultante.
    # -------------------------------------------------------------------------
    def degreeProduct(self, examplePos):
        degree = 1.0
        for i in range(self.antecedent.shape[0]):
            if degree > 0.0:
                degree *= self.dataBase.matching(self.antecedent[i,1], examplePos)
            else:
                break
        return degree * self.conf

    # -------------------------------------------------------------------------
    # NOMBRE: Setters / Getters
    # DESCRIPCIÓN: 
    #   Métodos estándar para gestionar confianza, soporte, WRACC y consecuente.
    # -------------------------------------------------------------------------
    def setConsequent(self, clas):
        self.clas = clas
    
    def setConfidence(self, conf):
        self.conf = conf

    def setSupport(self, supp):
        self.supp = supp

    def setWracc (self, wracc):
        self.wracc = wracc

    def getConfidence(self):
        return self.conf

    def getSupport(self):
        return self.supp

    def getWracc (self):
        return self.wracc

    def getClas (self):
        return self.clas

    # -------------------------------------------------------------------------
    # NOMBRE: isSubset
    # DESCRIPCIÓN: 
    #   Determina si la regla actual es un subconjunto de otra regla (poda).
    # ENTRADA:
    #   - rule [Rule]: Regla candidata a ser superconjunto.
    # SALIDA: 
    #   - [bool]: True si esta regla está contenida en la otra.
    # -------------------------------------------------------------------------
    def isSubset (self, rule):
        if (self.clas != rule.clas) or (self.nAnts > rule.nAnts):
            return False
        else:
            for k in range (self.antecedent.shape[0]):
                if self.antecedent[k,1] > -1:
                    if self.antecedent[k,0] != rule.antecedent[k,0]:
                        return False
            return True
    
    # -------------------------------------------------------------------------
    # NOMBRE: reduceWeight
    # DESCRIPCIÓN: 
    #   Evalúa el impacto de la regla sobre los pesos de los ejemplos y 
    #   cuenta cuántos ejemplos de su misma clase logra cubrir.
    # ENTRADA:
    #   - train [myDataSet]: Conjunto de entrenamiento.
    #   - exampleWeight [list]: Lista de estados de activación de los ejemplos.
    # SALIDA: 
    #   - count [int]: Número de ejemplos nuevos cubiertos.
    # -------------------------------------------------------------------------
    def reduceWeight (self, train, exampleWeight):
        count = 0
        for i in range(train.getnData()):
            ex = exampleWeight[i]
            if ex.isActive():
                if self.matching(i) > 0.0:##cambio
                    ex.incCount()
                    if (not ex.isActive()) and (train.getOutputAsInteger(i) == self.clas):
                        count += 1
        return count

    # -------------------------------------------------------------------------
    # NOMBRE: setLabel
    # DESCRIPCIÓN: 
    #   Establece o elimina una etiqueta difusa para una variable específica, 
    #   ajustando el contador de antecedentes activos.
    # ENTRADA:
    #   - pos [int]: Índice de la variable.
    #   - label [int]: Índice de la etiqueta difusa.
    # -------------------------------------------------------------------------
    def setLabel(self, pos, label):
        if (self.antecedent[pos,0] < 0) and (label > -1):
            self.nAnts += 1
        
        if (self.antecedent[pos,0] > -1) and (label < 0):
            self.nAnts -= 1
        self.antecedent[pos,0] = label

    # -------------------------------------------------------------------------
    # NOMBRE: __lt__ / __eq__
    # DESCRIPCIÓN: 
    #   Operadores de comparación basados en la métrica WRACC (Weighted Relative 
    #   Accuracy) para la ordenación de reglas.
    # -------------------------------------------------------------------------
    def __lt__(self, other):
        if other.wracc < self.wracc:
            return True
        elif other.wracc > self.wracc:
            return False

    def __eq__(self, other):
        return self.wracc == other.wracc
    

    # -------------------------------------------------------------------------
    # NOMBRE: __str__
    # DESCRIPCIÓN: 
    #   Devuelve la representación legible de la regla.
    #   Prioridad: Nombres Reales > Nombres DataBase > Genéricos.
    # -------------------------------------------------------------------------
    def __str__(self):
        text = "IF "
        connector = ""
        has_antecedents = False
        # Recorremos todas las variables
        for i in range(self.antecedent.shape[0]):
            label_index = int(self.antecedent[i, 0])
            
            # Si el índice es > -1, la variable participa en la regla
            if label_index > -1:
                # --- 1. RESOLUCIÓN DEL NOMBRE DE LA VARIABLE ---
                var_name = f"Var_{i}" # Valor por defecto
                # A) Intentamos usar nombres inyectados manualmente
                if hasattr(self, 'real_variable_names') and self.real_variable_names is not None and len(self.real_variable_names) > 0:
                    try:
                        var_name = self.real_variable_names[i]
                    except IndexError:
                        pass
                # B) Si no, intentamos usar nombres de la base de datos
                elif hasattr(self.dataBase, 'names'):
                    try:
                        var_name = self.dataBase.names[i]
                    except IndexError:
                        pass

                # --- 2. RESOLUCIÓN DEL NOMBRE DE LA ETIQUETA ---
                label_name = f"Label_{label_index}" # Valor por defecto

                # A) Intentamos usar nombres inyectados manualmente (Lista de Listas)
                if hasattr(self, 'real_label_names') and self.real_label_names is not None and len(self.real_label_names) > 0:
                    try:
                        # Accedemos a la lista de la variable i, y luego a la etiqueta
                        label_name = self.real_label_names[i][label_index]
                    except (IndexError, TypeError):
                        pass
                # B) Si no, intentamos usar el método print() de DataBase
                elif hasattr(self.dataBase, 'print'):
                    try:
                        label_name = self.dataBase.print(i, label_index)
                    except (AttributeError, IndexError):
                        pass

                # Construcción del fragmento
                text += f"{connector}({var_name} IS {label_name})"
                connector = " AND "
                has_antecedents = True
        
        if not has_antecedents:
            text += "(Empty Rule)"

        # --- 3. RESOLUCIÓN DEL NOMBRE DE LA CLASE ---
        class_index = int(self.clas)
        class_name = str(class_index) # Valor por defecto

        # A) Intentamos usar nombres inyectados manualmente
        if hasattr(self, 'real_class_names') and self.real_class_names is not None and len(self.real_class_names) > 0:
            try:
                class_name = self.real_class_names[class_index]
            except IndexError:
                pass
        # B) Si no, intentamos buscar en target_names de la DataBase
        else:
            try:
                # Usamos getattr para seguridad por si target_names no existe
                target_names = getattr(self.dataBase, 'target_names', [])
                class_name = target_names[class_index]
            except (IndexError, TypeError):
                pass

        text += f" THEN Class: {class_name} (RW: {self.conf:.4f})"
        
        return text
    def setRealVariableNames(self, variables):
        self.real_variable_names = variables
    
    def setRealClassNames(self, classes):
        self.real_class_names = classes