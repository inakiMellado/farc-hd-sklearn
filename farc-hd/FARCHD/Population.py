# Title: Population
# Description: This class contains the population for the CHC algorithm
# Copyright: KEEL Copyright (c) 2010
# Company: KEEL 
# @author Written by Jesus Alcal� (University of Granada) 09/02/2010
# @version 1.2
# @since JDK1.5

# JAva to python translator: Iñaki Mellado Ilundain

import numpy as np
from farc_hd.FARCHD.DataBase import DataBase
from farc_hd.FARCHD.RuleBase import RuleBase
from farc_hd.FARCHD.Individual import Individual
from farc_hd.FARCHD.myDataSetV2 import myDataSet
from farc_hd.org.core.Randomize import Randomize
import time
class Population:
	# -------------------------------------------------------------------------
    # NOMBRE: __init__
    # DESCRIPCIÓN: 
    #   Constructor de la población. Configura el tamaño, los parámetros del 
    #   algoritmo CHC y las estructuras para almacenar individuos.
    # ENTRADA:
    #   - train: Dataset de entrenamiento.
    #   - dataBase: Base de datos difusa.
    #   - ruleBase: Base de reglas inicial.
    #   - size [int]: Tamaño de la población.
    #   - BITS_GEN [int]: Precisión binaria de los genes reales.
    #   - maxTrials [int]: Número máximo de evaluaciones permitidas.
    #   - alpha [float]: Peso para el cálculo del fitness.
    # -------------------------------------------------------------------------
	def __init__ (self, train, dataBase, ruleBase, size, BITS_GEN, maxTrials, alpha):
		if train is not None:
			self.dataBase = dataBase
			self.train = train
			self.ruleBase = ruleBase
			self.BITS_GEN = BITS_GEN

			self.n_variables = dataBase.numVariables()
			self.pop_size = size
			self.alpha = alpha
			self.maxTrials = maxTrials
			self.Lini = ((dataBase.getnLabelsReal() * BITS_GEN) + ruleBase.size()) / 4.0
			self.L = self.Lini
			self.w1 = self.alpha * ruleBase.size()
			self.numReiniciosSinMejora = 0
			self.bestFitness = 0.0

			self.Population = []
			self.selected = np.zeros(self.pop_size, dtype = int)
			self.evolution = ""
	# -------------------------------------------------------------------------
    # NOMBRE: BETTER
    # DESCRIPCIÓN: 
    #   determina si a > b o al reves
    # -------------------------------------------------------------------------
	def BETTER(self, a, b):
		if a > b:
			return True
		return False
	
	# -------------------------------------------------------------------------
    # NOMBRE: Generation
    # DESCRIPCIÓN: 
    #   Bucle principal del algoritmo evolutivo. Ejecuta ciclos de selección, 
    #   cruce, evaluación y elitismo hasta agotar las evaluaciones o reinicios.
    # -------------------------------------------------------------------------
	def Generation(self):
		self.init()
		self.evaluate(0)

		tipoCruce = 0
		
		while self.nTrials < self.maxTrials and self.numReiniciosSinMejora < 3:
			self.selection()
			self.crossover()
			self.evaluate(self.pop_size)
			self.elitist()
			if not self.hasNew():
				self.L -= 30
				if self.L < 0.0:
					self.restart()
					if self.Population[0].fitness <= self.bestFitness:
						self.numReiniciosSinMejora += 1
					else:
						self.bestFitness = self.Population[0].fitness
						self.numReiniciosSinMejora = 0

	# -------------------------------------------------------------------------
    # NOMBRE: init
    # DESCRIPCIÓN: 
    #   Crea la población inicial. El primer individuo es una copia "limpia" 
    #   del sistema inicial; el resto se genera aleatoriamente.
    # -------------------------------------------------------------------------
	def init(self):
		ind = Individual(self.ruleBase, self.dataBase, self.w1)
		ind.reset()
		self.Population.append(ind)
		for i in range (1, self.pop_size):
			ind = Individual(self.ruleBase, self.dataBase, self.w1)
			ind.randomValues()
			self.Population.append(ind)
		self.best_fitness = 0.0
		self.nTrials = 0
		
	# -------------------------------------------------------------------------
    # NOMBRE: evaluate
    # DESCRIPCIÓN: 
    #   Calcula el fitness de los individuos a partir de una posición dada.
    # ENTRADA:
    #   - pos [int]: Índice inicial en la lista de la población para evaluar.
    # -------------------------------------------------------------------------
	def evaluate(self, pos):
		for i in range(pos, len(self.Population)):
			self.Population[i].evaluate()

		self.nTrials += (len(self.Population) - pos)
	
	# -------------------------------------------------------------------------
    # NOMBRE: selection
    # DESCRIPCIÓN: 
    #   Realiza una selección aleatoria de la población barajando los índices 
    #   de los individuos. Prepara el emparejamiento para el cruce.
    # -------------------------------------------------------------------------
	def selection(self):
		for i in range (self.pop_size):
			self.selected[i] = i

		for i in range (self.pop_size):
			random = Randomize.Randint(0, self.pop_size)
			self.selected[random], self.selected[i] = self.selected[i], self.selected[random]

	# -------------------------------------------------------------------------
    # NOMBRE: xPC_BLX
    # DESCRIPCIÓN: 
    #   Invoca el operador de cruce BLX-alpha entre dos individuos hijos
    #   para la sintonización de los parámetros de los conjuntos difusos.
    # ENTRADA:
    #   - d [float]: Valor de alpha para el rango del cruce.
    #   - son1 [Individual]: Primer descendiente.
    #   - son2 [Individual]: Segundo descendiente.
    # SALIDA: 
    #   - None
    # -------------------------------------------------------------------------
	def xPC_BLX(self, d, son1, son2):
		son1.xPC_BLX(son2, d)
	
	# -------------------------------------------------------------------------
    # NOMBRE: twoPoint
    # DESCRIPCIÓN: 
    #   Aplica el cruce de dos puntos entre los cromosomas de dos descendientes.
    # ENTRADA:
    #   - son1 [Individual]: Primer descendiente.
    #   - son2 [Individual]: Segundo descendiente.

    # -------------------------------------------------------------------------
	def twoPoint(self, son1, son2):
		son1.twoPoint(son2)
	
	# -------------------------------------------------------------------------
    # NOMBRE: Hux
    # DESCRIPCIÓN: 
    #   Aplica el cruce HUX para la parte binaria de los cromosomas.
    # ENTRADA:
    #   - son1 [Individual]: Primer descendiente.
    #   - son2 [Individual]: Segundo descendiente.
    # -------------------------------------------------------------------------
	def Hux(self, son1, son2):
		son1.Hux(son2)
	
	# -------------------------------------------------------------------------
    # NOMBRE: crossover
    # DESCRIPCIÓN: 
    #   Aplica los operadores de cruce (BLX para reales y HUX para binarios) 
    #   solo si la distancia entre padres supera el umbral L actual.
    # -------------------------------------------------------------------------
	def crossover(self):
		for i in range(0, self.pop_size, 2):
			dad = self.Population[self.selected[i]]
			mom = self.Population[self.selected[i + 1]]
			dist = dad.distHamming(mom, self.BITS_GEN) / 2.0
			if dist > self.L:
				son1 = dad.clone()
				son2 = mom.clone()
				self.xPC_BLX(1.0, son1, son2)
				self.Hux(son1, son2)
				son1.onNew()
				son2.onNew()
				self.Population.append(son1)
				self.Population.append(son2)
	
	# -------------------------------------------------------------------------
    # NOMBRE: elitist
    # DESCRIPCIÓN: 
    #   Mantiene el tamaño de la población seleccionando a los N mejores 
    #   individuos (padres + hijos) y descarta el resto.
    # -------------------------------------------------------------------------
	def elitist(self):
		self.Population.sort()
		while len(self.Population) > self.pop_size:
			self.Population.pop(self.pop_size)
		self.best_fitness = self.Population[0].getFitness()
		self.evolution += "Accuracy / Fitness in the evaluacion " + str(self.nTrials) + ": " + str(self.Population[0].getAccuracy()) + " / " + str(self.best_fitness) + "\n"

		# print("Accuracy / Fitness in the evaluacion " + str(self.nTrials) + ": " +  str(self.Population[0].getAccuracy()) + " / " + str(self.best_fitness) )
	
	# -------------------------------------------------------------------------
    # NOMBRE: getEvolution
    # DESCRIPCIÓN: 
    #   Recupera el historial acumulado del rendimiento (Accuracy/Fitness).
    # SALIDA: 
    #   - [str]: Cadena de texto con el log de la evolución.
    # -------------------------------------------------------------------------
	def getEvolution(self):
		return self.evolution
	
	# -------------------------------------------------------------------------
    # NOMBRE: hasNew
    # DESCRIPCIÓN: 
    #   Verifica si han entrado nuevos individuos a la población en la 
    #   generación actual y resetea sus estados de "nuevo".
    # SALIDA: 
    #   - state [bool]: True si hay individuos nuevos, False de lo contrario.
    # -------------------------------------------------------------------------
	def hasNew(self):
		state = False
		for i in range (self.pop_size):
			ind = self.Population[i]
			if ind.isNew():
				ind.offNew()
				state = True
		return state
	
	#-------------------------------------------------------------------------
    # NOMBRE: restart
    # DESCRIPCIÓN: 
    #   Realiza un reinicio de la población cuando el umbral L cae bajo cero. 
    #   Mantiene al mejor individuo y genera el resto aleatoriamente.
    # -------------------------------------------------------------------------
	def restart(self):
		self.w1 = 0.0
		self.Population.sort()
		ind = self.Population[0].clone()
		ind.setw1(self.w1)

		self.Population.clear()
		self.Population.append(ind)

		for i in range (1, self.pop_size):
			ind = Individual(self.ruleBase, self.dataBase, self.w1)
			ind.randomValues()
			self.Population.append(ind)
				

		self.evaluate(0)
		## self.evaluate(1)
		self.L = self.Lini
	
	# -------------------------------------------------------------------------
    # NOMBRE: getBestRB
    # DESCRIPCIÓN: 
    #   Obtiene la mejor Base de Reglas física generada por el mejor 
    #   individuo de la población actual.
    # SALIDA:
    #   - ruleBase [RuleBase]: La base de reglas optimizada final.
    # -------------------------------------------------------------------------
	def getBestRB(self):
		self.Population.sort()
		ruleBase = self.Population[0].generateRB()
		return ruleBase