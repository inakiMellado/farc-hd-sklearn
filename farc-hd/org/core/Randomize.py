from farc_hd.org.core.MTwister import MTwister

class Randomize:
    # Instanciamos la nueva clase JIT compilada
    generador = MTwister()

    # -------------------------------------------------------------------------
    # NOMBRE: setSeed
    # DESCRIPCIÓN: 
    #   Establece la semilla inicial para el generador Mersenne Twister. 
    #   Garantiza la reproducibilidad de los experimentos.
    # ENTRADA:
    #   - semilla [uint32]: Valor numérico para inicializar el estado.
    # -------------------------------------------------------------------------
    @staticmethod
    def setSeed(semilla):
        Randomize.generador.init_genrand(semilla)

    # -------------------------------------------------------------------------
    # NOMBRE: Rand
    # DESCRIPCIÓN: 
    #   Genera un número real aleatorio en el intervalo [0, 1) con resolución 
    #   de 53 bits (precisión doble).
    # SALIDA: 
    #   - [float64]: Valor aleatorio uniforme.
    # -------------------------------------------------------------------------
    @staticmethod
    def Rand():
        return Randomize.generador.genrand_res53()

    # -------------------------------------------------------------------------
    # NOMBRE: RandOpen
    # DESCRIPCIÓN: 
    #   Genera un número real aleatorio en el intervalo abierto (0, 1).
    # SALIDA: 
    #   - [float64]: Valor aleatorio uniforme excluyendo 0 y 1.
    # -------------------------------------------------------------------------
    @staticmethod
    def RandOpen():
        return Randomize.generador.genrand_real3()
    
    # -------------------------------------------------------------------------
    # NOMBRE: RandClosed
    # DESCRIPCIÓN: 
    #   Genera un número real aleatorio en el intervalo cerrado [0, 1].
    # SALIDA: 
    #   - [float64]: Valor aleatorio uniforme incluyendo 0 y 1.
    # -------------------------------------------------------------------------
    @staticmethod
    def RandClosed():
        return Randomize.generador.genrand_real1()
    
    # -------------------------------------------------------------------------
    # NOMBRE: RandGaussian
    # DESCRIPCIÓN: 
    #   Genera un valor con distribución normal (Gaussiana) mediante la 
    #   aproximación del Teorema del Límite Central.
    # SALIDA: 
    #   - [float64]: Valor con distribución N(0, 1) aproximada.
    # -------------------------------------------------------------------------
    @staticmethod
    def RandGaussian():
        return Randomize.generador.genrand_gaussian()

    # -------------------------------------------------------------------------
    # NOMBRE: Randint
    # DESCRIPCIÓN: 
    #   Genera un número entero aleatorio en el rango [low, high).
    # ENTRADA:
    #   - low [int]: Límite inferior (incluido).
    #   - high [int]: Límite superior (excluido).
    # SALIDA: 
    #   - [int]: Entero aleatorio escalado.
    # -------------------------------------------------------------------------
    @staticmethod
    def Randint(low, high):
        return int((low + (high - low) * Randomize.generador.genrand_res53()))
    
    # -------------------------------------------------------------------------
    # NOMBRE: RandintOpen
    # DESCRIPCIÓN: 
    #   Genera un número entero aleatorio en el rango abierto (low, high).
    # ENTRADA:
    #   - low [int]: Límite inferior (excluido).
    #   - high [int]: Límite superior (excluido).
    # SALIDA: 
    #   - [int]: Entero aleatorio.
    # -------------------------------------------------------------------------
    @staticmethod
    def RandintOpen(low, high):
        return int(((low+1) + (high - (low+1)) * Randomize.generador.genrand_res53()))
    
    # -------------------------------------------------------------------------
    # NOMBRE: RandintClosed
    # DESCRIPCIÓN: 
    #   Genera un número entero aleatorio en el rango cerrado [low, high].
    # ENTRADA:
    #   - low [int]: Límite inferior (incluido).
    #   - high [int]: Límite superior (incluido).
    # SALIDA: 
    #   - [int]: Entero aleatorio.
    # -------------------------------------------------------------------------
    @staticmethod
    def RandintClosed(low, high):
        return int((low + ((high + 1) - low) * Randomize.generador.genrand_res53()))
    
    # -------------------------------------------------------------------------
    # NOMBRE: Randdouble
    # DESCRIPCIÓN: 
    #   Genera un número real aleatorio escalado al rango [low, high).
    # ENTRADA:
    #   - low [float]: Límite inferior.
    #   - high [float]: Límite superior.
    # SALIDA: 
    #   - [float64]: Valor real escalado.
    # -------------------------------------------------------------------------
    @staticmethod
    def Randdouble(low, high):
        return (low + (high-low) * Randomize.generador.genrand_res53())
    
    # -------------------------------------------------------------------------
    # NOMBRE: RanddoubleOpen
    # DESCRIPCIÓN: 
    #   Genera un número real aleatorio escalado al rango abierto (low, high).
    # ENTRADA:
    #   - low [float]: Límite inferior.
    #   - high [float]: Límite superior.
    # SALIDA: 
    #   - [float64]: Valor real escalado.
    # -------------------------------------------------------------------------
    @staticmethod
    def RanddoubleOpen(low, high):
        return (low + (high-low) * Randomize.generador.genrand_real3())
    
    # -------------------------------------------------------------------------
    # NOMBRE: RanddoubleClosed
    # DESCRIPCIÓN: 
    #   Genera un número real aleatorio escalado al rango cerrado [low, high].
    # ENTRADA:
    #   - low [float]: Límite inferior.
    #   - high [float]: Límite superior.
    # SALIDA: 
    #   - [float64]: Valor real escalado.
    # -------------------------------------------------------------------------
    @staticmethod
    def RanddoubleClosed(low, high):
        return (low + (high-low) * Randomize.generador.genrand_real1())
    
    # -------------------------------------------------------------------------
    # NOMBRE: getGenerator
    # DESCRIPCIÓN: 
    #   Proporciona acceso directo a la instancia del generador MTwister. 
    #   Esencial para pasar el estado del generador a funciones optimizadas 
    #   con Numba que requieren una "jitclass".
    # SALIDA: 
    #   - [MTwister]: Instancia compilada del generador.
    # -------------------------------------------------------------------------
    @staticmethod
    def getGenerator():
        return Randomize.generador