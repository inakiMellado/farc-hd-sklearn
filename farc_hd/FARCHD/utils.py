import random
import numpy as np
import math
from operator import attrgetter
from deap import tools

# Nombre funcion: bin2gray
# Descripcion:
#   Convierte una lista de bits en formato binario estándar a código Gray.
# Explicaciones de las variables de entrada:
#   - bits [list]: Lista de enteros (0 o 1) que representan un número en binario.
# Explicacion de las variables de salida:
#   - [list]: Lista de bits codificados en Gray.
def bin2gray(bits):
    return bits[:1] + [i ^ ishift for i, ishift in zip(bits[:-1], bits[1:])]

# Nombre funcion: gray
# Descripcion:
#   Escala un número real a un entero, lo convierte a binario y devuelve su código Gray con longitud fija.
# Explicaciones de las variables de entrada:
#   - numero [float]: Valor real a codificar.
#   - numBits [int]: Cantidad de bits deseada para la representación.
#   - minimo [float]: Límite inferior del rango del número.
#   - maximo [float]: Límite superior del rango del número.
# Explicacion de las variables de salida:
#   - codigoGray [list]: Lista de bits que representan el valor en código Gray.
def gray(numero, numBits, minimo, maximo):

    ent = int((((numero - minimo) * (math.pow(2, numBits) - 1)) / (maximo - minimo)) + 0.5)

    binario = []
    while ent>=1:
        bina = ent%2
        ent = ent/2
        binario.append(bina)
    binario.reverse()

    codigoGray = bin2gray(binario)
    if len(codigoGray)<numBits:
        codigoGray = [0]*(numBits-len(codigoGray)) + codigoGray

    return codigoGray

# Nombre funcion: distanciaHamming
# Descripcion:
#   Calcula la cantidad de posiciones en las que dos individuos difieren (distancia de Hamming).
# Explicaciones de las variables de entrada:
#   - ind1 [list/array]: Primer individuo o cadena de bits.
#   - ind2 [list/array]: Segundo individuo o cadena de bits.
# Explicacion de las variables de salida:
#   - [int]: Número de bits o elementos diferentes.    
def distanciaHamming(ind1, ind2):
    return(sum(np.array(ind1)!=np.array(ind2)))

# Nombre funcion: checkBounds
# Descripcion:
#   Decorador para asegurar que los hijos generados por una función se mantengan dentro de los límites definidos.
# Explicaciones de las variables de entrada:
#   - min [float]: Valor mínimo permitido para los genes.
#   - max [float]: Valor máximo permitido para los genes.
# Explicacion de las variables de salida:
#   - decorator [function]: Función decoradora que envuelve la lógica de cruce o mutación.
def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in xrange(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring
        return wrapper
    return decorator

# Nombre funcion: My_cxBlend
# Descripcion:
#   Implementa el cruce BLX-alpha (Blend Crossover) estándar para individuos reales.
# Explicaciones de las variables de entrada:
#   - ind1 [list]: Cromosoma del primer padre.
#   - ind2 [list]: Cromosoma del segundo padre.
#   - alpha [float]: Factor de extensión para el rango de descendencia.
# Explicacion de las variables de salida:
#   - ind1, ind2 [tuple]: Los dos individuos tras aplicar el cruce.
def My_cxBlend(ind1, ind2, alpha):
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        gamma = (1. + 2. * alpha) * random.random() - alpha
        ind1[i] = (1. - gamma) * x1 + gamma * x2
        gamma = (1. + 2. * alpha) * random.random() - alpha
        ind2[i] = (1. - gamma) * x1 + gamma * x2

    return ind1, ind2

# Nombre funcion: My_pcxBlend
# Descripcion:
#   Variante del cruce BLX donde el rango de descendencia se calcula de forma independiente para cada hijo basándose en la distancia entre padres.
# Explicaciones de las variables de entrada:
#   - ind1 [list]: Cromosoma del primer padre.
#   - ind2 [list]: Cromosoma del segundo padre.
#   - alpha [float]: Factor de escala para el rango de búsqueda.
# Explicacion de las variables de salida:
#   - ind1, ind2 [tuple]: Los dos individuos modificados.
def My_pcxBlend(ind1, ind2, alpha):
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        rango = abs(x1-x2)
        minimo = x1 - rango * alpha
        maximo = x1 + rango * alpha
        ind1[i] = minimo+random.random()*(maximo-minimo)
        minimo = x2 - rango * alpha
        maximo = x2 + rango * alpha
        ind2[i] = minimo+random.random()*(maximo-minimo)

    return ind1, ind2

# Nombre funcion: hux
# Descripcion:
#   Implementa el cruce Half Uniform (HUX). Intercambia exactamente la mitad de los bits diferentes entre los padres.
# Explicaciones de las variables de entrada:
#   - ind1 [list]: Primer individuo binario.
#   - ind2 [list]: Segundo individuo binario.
# Explicacion de las variables de salida:
#   - ind1, ind2 [tuple]: Los dos individuos con los bits intercambiados.
def hux(ind1, ind2):
    indices = [i for i in range(len(ind1)) if ind1[i] != ind2[i]]
    longitud = len(indices)/2
    np.random.shuffle(indices)
    for i in indices[:longitud]:
        ind1[i]=ind2[i]
    for i in indices[longitud:]:
        ind2[i]=ind1[i]

    return ind1, ind2

# Nombre funcion: selLineal
# Descripcion:
#   Selecciona k individuos mediante el método de la ruleta lineal, adaptado para problemas de minimización.
# Explicaciones de las variables de entrada:
#   - individuals [list]: Lista de individuos de los cuales seleccionar.
#   - k [int]: El número de individuos a seleccionar.
# Explicacion de las variables de salida:
#   - chosen [list]: Lista de individuos seleccionados.
def selLineal(individuals, k):
    """Selecciona k individuos por el metodo de la ruleta lineal, programada para minimización

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.

    """
    popWithIndex = sorted(enumerate(individuals), key=lambda x: x[1].fitness.values[0], reverse=True)
    (indices, pop) = zip(*popWithIndex)
    N = len(pop)

    chosen = []
    for i in xrange(k):
        u = random.random()
        sum_ = 0.0
        for j, ind in enumerate(pop):
            sum_ += float(j+1)/float((N*(N+1)/2))
            if sum_ > u:
                chosen.append(ind)
                break

    return chosen

# Nombre funcion: selRouletteMinimization
# Descripcion:
#   Selecciona k individuos mediante el método de la ruleta proporcional, modificado para minimización usando la inversa del fitness.
# Explicaciones de las variables de entrada:
#   - individuals [list]: Población de la cual seleccionar.
#   - k [int]: Cantidad de individuos a elegir.
# Explicacion de las variables de salida:
#   - chosen [list]: Lista de individuos seleccionados.
def selRouletteMinimization(individuals, k):
    """Select *k* individuals from the input *individuals* using *k*
    spins of a roulette. 
    
    Funcion modificada para que sirva para minimizar haciendo la inversa del fitness

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.

    """
    s_inds = sorted(individuals, key=attrgetter("fitness"), reverse=True)
    sum_fits = sum(1.0/ind.fitness.values[0] for ind in individuals)

    chosen = []
    for i in xrange(k):
        u = random.random() * sum_fits
        sum_ = 0
        for ind in s_inds:
            sum_ += 1.0/ind.fitness.values[0]
            if sum_ > u:
                chosen.append(ind)
                break

    return chosen

# Nombre funcion: replacement
# Descripcion:
#   Gestiona el reemplazo generacional de la población, permitiendo elitismo simple.
# Explicaciones de las variables de entrada:
#   - pop [list]: Población actual.
#   - offspring [list]: Población descendiente.
#   - n [int]: Número de mejores individuos a mantener (elitismo).
#   - maximizar [bool]: Define si el objetivo es maximizar o minimizar.
# Explicacion de las variables de salida:
#   - pop [list]: Nueva población para la siguiente generación.
def replacement(pop, offspring, n=0, maximizar=False):
    # Reemplazamos la población completa (generacional sin elitismo) # 628.10
    if (n==0):
        pop = offspring
    else:
    # Reemplazamos la población completa y metemos el mejor: elitismo #480.28
        best_ind = sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=maximizar)[:n]
        pop[:-n] = sorted(offspring, key=lambda ind: ind.fitness.values[0], reverse=maximizar)[:-n]
        pop[-n:] = best_ind
    return pop

# Nombre funcion: replacementMaximoElitismo
# Descripcion:
#   Wrapper para realizar un reemplazo elitista total utilizando el esquema CHC.
# Explicaciones de las variables de entrada:
#   - pop [list]: Población original.
#   - offspring [list]: Población de hijos.
#   - maximizar [bool]: Indica si el problema es de maximización.
# Explicacion de las variables de salida:
#   - pop [list]: Población combinada y filtrada con los mejores individuos.
def replacementMaximoElitismo(pop, offspring, maximizar=False):
    # Padres e hijos compiten para formar parte de la siguietne generacion
    pop, aux = replacementCHC(pop, offspring, maximizar)
    return pop

# Nombre funcion: replacementCHC
# Descripcion:
#   Implementa el reemplazo del algoritmo CHC, donde padres e hijos compiten por los N mejores puestos y detecta si hubo mejora.
# Explicaciones de las variables de entrada:
#   - pop [list]: Población actual.
#   - offspring [list]: Descendencia generada.
#   - maximizar [bool]: Define el orden de la selección (mejor fitness).
# Explicacion de las variables de salida:
#   - pop [list]: Población resultante de tamaño N.
#   - pasanHijos [bool]: True si al menos un descendiente entró en la nueva población.
def replacementCHC(pop, offspring, maximizar=False):
    # Padres e hijos compiten para formar parte de la siguietne generacion
    popWithIndex = sorted(enumerate(pop + offspring), key=lambda x: x[1].fitness.values[0], reverse=maximizar)[:len(pop)]
    (indicesSelected, pop) = zip(*popWithIndex)
    pop = list(pop)
    hijosQuePasan = len(filter(lambda x: x >= len(pop), indicesSelected))
    pasanHijos = True
    if (hijosQuePasan==0):
        pasanHijos = False
    return pop, pasanHijos