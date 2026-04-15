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

from farc_hd.FARCHD.RuleBase import RuleBase
from farc_hd.FARCHD.Itemset import Itemset
from farc_hd.FARCHD.Item import Item

class Apriori:
    # -------------------------------------------------------------------------
    # NOMBRE: generateRB
    # DESCRIPCIÓN: 
    #   calcula los soportes mínimos por clase.
    # Variables de entrada:
    #   - ruleBase: Objeto base donde se almacenarán las reglas finales.
    #   - dataBase: Base de datos con la definición de etiquetas y variables.
    #   - train: Conjunto de datos de entrenamiento.
    #   - minsup: Valor de soporte mínimo relativo.
    #   - minconf: Valor de confianza mínima para la generación de reglas.
    #   - depth: Profundidad máxima (número de ítems) de los itemsets a generar.
    # -------------------------------------------------------------------------
    def __init__(self, ruleBase=None, dataBase=None, train=None, minsup=None, minconf=None, depth=None):
        self.train = train
        self.dataBase = dataBase
        self.ruleBase = ruleBase
        self.minconf = minconf
        self.depth = depth
        self.nClasses = self.train.getnClasses()
        self.nVariables = self.train.getnInputs()

        self.L2 = []
        self.minSupps = [self.train.frecuentClass(i) * minsup for i in range(self.nClasses)]

    # -------------------------------------------------------------------------
    # NOMBRE: generateRB
    # DESCRIPCIÓN: 
    #   Itera sobre cada clase para generar itemsets frecuentes, reglas y reducir el conjunto final de reglas.
    # -------------------------------------------------------------------------
    def generateRB(self):
        self.ruleStage1 = 0
        self.ruleBaseClase = RuleBase(self.dataBase, self.train, self.ruleBase.getK(), self.ruleBase.getTypeInference())
        
        for i in range(self.nClasses):
            self.minsup = self.minSupps[i]
            self.generateL2(i)
            self.generateLarge(self.L2, i)
            self.ruleBaseClase.reduceRules(i)

            self.ruleBase.add(self.ruleBaseClase)
            self.ruleBaseClase.clear()
            
        self.ruleBase.almacenaPesos()

    # -------------------------------------------------------------------------
    # NOMBRE: generateL2
    # DESCRIPCIÓN: 
    #   Filtra y genera itemsets de nivel 1 (un solo ítem) que superan el 
    #   umbral de soporte mínimo para una clase específica.
    # ENTRADA:
    #   - clas [int]: Índice de la clase objetivo.
    # -------------------------------------------------------------------------
    def generateL2(self, clas):
        self.L2 = []
        itemset = Itemset(clas)
        absValue = 0
        for i in range(self.nVariables):
            if self.dataBase.numLabels(i) > 1:
                for j in range(self.dataBase.numLabels(i)):
                    item = Item(i, j, absValue)
                    itemset.add(item)
                    itemset.calculateSupports(self.dataBase)
                    
                    absValue += 1
                    if itemset.getSupportClass() >= self.minsup:
                        self.L2.append(itemset.clone())
                    itemset.remove(0)

        self.generateRules(self.L2, clas)

    # -------------------------------------------------------------------------
    # NOMBRE: hasUncoverClass
    # DESCRIPCIÓN: 
    #   Verifica el número de ejemplos de una clase que no han sido 
    #   cubiertos por los itemsets actuales.
    # ENTRADA:
    #   - clas [int]: Índice de la clase a verificar.
    # SALIDA: 
    #   - uncover [int]: Cantidad de ejemplos sin cobertura.
    # -------------------------------------------------------------------------
    def hasUncoverClass(self, clas):
        uncover = 0
        for j in range(self.train.size()):
            if self.train.getOutputAsInteger(j) == clas:
                stop = False
                for itemset in self.L2:
                    degree = itemset.degree(self.dataBase, self.train.getExample(j))
                    if degree > 0.0:
                        stop = True
                        break
                if not stop:
                    uncover += 1
        return uncover

    # -------------------------------------------------------------------------
    # NOMBRE: generateLarge
    # DESCRIPCIÓN: 
    #   Proceso recursivo que expande itemsets (k -> k+1) combinando
    #   elementos combinables que mantienen el soporte mínimo.
    # ENTRADA:
    #   - Lk [list]: Lista de itemsets del nivel actual.
    #   - clas [int]: Índice de la clase analizada.
    # SALIDA: 
    #   - None
    # -------------------------------------------------------------------------
    def generateLarge(self, Lk, clas):
        size = len(Lk)
        if size > 1:
            if Lk[0].size() < self.nVariables and Lk[0].size() < self.depth:
                Lnew = []
                for i in range(size - 1):
                    itemseti = Lk[i]
                    for j in range(i + 1, size):
                        itemsetj = Lk[j]
                        
                        if self.isCombinable(itemseti, itemsetj):
                            newItemset = itemseti.clone()
                            newItemset.add((itemsetj.get(itemsetj.size()-1)).clone())
                            newItemset.calculateSupports(self.dataBase) ## Cambio
                            if newItemset.getSupportClass() >= self.minsup:
                                Lnew.append(newItemset)

                    self.generateRules(Lnew, clas)
                    self.generateLarge(Lnew, clas)
                    Lnew.clear()

    # -------------------------------------------------------------------------
    # NOMBRE: isCombinable
    # DESCRIPCIÓN: 
    #   Lógica de poda para asegurar que solo se combinen itemsets siguiendo una 
    #   secuencia indexada ascendente de variables
    # ENTRADA:
    #   - itemseti: Primer conjunto de ítems.
    #   - itemsetj: Segundo conjunto de ítems.
    # SALIDA: 
    #   - [bool]: True si son combinables, False en caso contrario.
    # -------------------------------------------------------------------------
    def isCombinable(self, itemseti, itemsetj):
        itemi = itemseti.get(itemseti.size()-1)
        itemj = itemsetj.get(itemseti.size()-1)
        return itemi.getVariable() < itemj.getVariable()

    # -------------------------------------------------------------------------
    # NOMBRE: getRulesStage1
    # DESCRIPCIÓN: 
    #   Retorna el conteo acumulado de reglas encontradas en la primera etapa.
    # SALIDA: 
    #   - [int]: Total de reglas generadas.
    # -------------------------------------------------------------------------
    def getRulesStage1(self):
        return self.ruleStage1

    # -------------------------------------------------------------------------
    # NOMBRE: generateRules
    # DESCRIPCIÓN: 
    #   Calcula la confianza de los itemsets y decide cuáles se integran
    #   como reglas y cuáles se eliminan del proceso de expansión.
    # ENTRADA:
    #   - Lk [list]: Lista de itemsets candidatos.
    #   - clas [int]: Índice de la clase objetivo.
    # -------------------------------------------------------------------------
    def generateRules(self, Lk, clas):
        i = len(Lk) - 1
        while i >= 0:
            itemset = Lk[i]
            if itemset.getSupport() > 0.0:
                confidence = itemset.getSupportClass() / itemset.getSupport()
            else:
                confidence = 0.0
            if confidence > 0.4:
                self.ruleBaseClase.add(itemset)
                self.ruleStage1 += 1
            if confidence > self.minconf:
                Lk.pop(i)
            i -= 1