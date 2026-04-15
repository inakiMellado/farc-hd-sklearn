import numpy as np
from numba import int32, float64, uint32, njit
from numba.experimental import jitclass

# Definimos la estructura de tipos para Numba
spec = [
    ('state', uint32[:]), # Usamos uint32 para replicar el comportamiento de mascara 0xffffffff
    ('left', int32),
    ('initf', int32),
    ('inext', int32),
]

@jitclass(spec)
class MTwister:
    # -------------------------------------------------------------------------
    # NOMBRE: __init__
    # DESCRIPCIÓN: 
    #   Constructor de la clase MTwister. Inicializa el vector de estado y los 
    #   punteros necesarios para el algoritmo Mersenne Twister.
    # SALIDA: 
    #   - None
    # -------------------------------------------------------------------------
    def __init__(self):
        self.state = np.zeros(624, dtype=np.uint32)
        self.left = 1
        self.initf = 0
        self.inext = 0
        
        # Inicialización por defecto si no se llama a nada mas (aunque se suele llamar a init_genrand)
        self.left = 1
        self.initf = 0

    # -------------------------------------------------------------------------
    # NOMBRE: init_genrand
    # DESCRIPCIÓN: 
    #   Inicializa el vector de estado del generador utilizando una semilla entera. 
    #   Aplica una secuencia aritmética para rellenar las 624 palabras de estado.
    # ENTRADA:
    #   - s [uint32]: Valor de la semilla (seed) para la inicialización.
    # SALIDA: 
    #   - None
    # -------------------------------------------------------------------------
    def init_genrand(self, s):
        self.state[0] = s & 0xffffffff
        for j in range(1, 624):
            # Lógica exacta del archivo original:
            # (1812433253 * (self.state[j - 1] ^ (self.state[j - 1] >> 30)) + j)
            prev = self.state[j-1]
            xor_val = prev ^ (prev >> 30)
            # En numba uint32, la multiplicacion puede hacer overflow, eso es el comportamiento deseado 
            # equivalente a & 0xffffffff
            val = (1812433253 * xor_val + j)
            self.state[j] = val & 0xffffffff
            
        self.left = 1
        self.initf = 1

    # -------------------------------------------------------------------------
    # NOMBRE: next_state
    # DESCRIPCIÓN: 
    #   Genera un nuevo bloque de 624 palabras de estado mediante el proceso 
    #   de "twist" (recombinación de bits). Se activa cuando el pool actual 
    #   de números aleatorios se agota.
    # SALIDA: 
    #   - None (Modifica el array interno self.state)
    # -------------------------------------------------------------------------
    def next_state(self):
        p = 0
        # No necesitamos init automatico con 5489 si asumimos que el usuario siempre inicializa
        # Pero por seguridad mantenemos la logica si initf es 0
        if self.initf == 0:
            self.init_genrand(5489)
            
        self.left = 624
        self.inext = 0
        
        # Bucles originales convertidos a indices directos para Numba
        for j in range(227): # 624 - 397
            y = (self.state[p] & 0x80000000) | (self.state[p + 1] & 0x7fffffff)
            if (self.state[p + 1] & 1) != 0:
                 self.state[p] = self.state[p + 397] ^ (y >> 1) ^ 0x9908b0df
            else:
                 self.state[p] = self.state[p + 397] ^ (y >> 1)
            p += 1

        for j in range(396): # 397 - 1
            y = (self.state[p] & 0x80000000) | (self.state[p + 1] & 0x7fffffff)
            if (self.state[p + 1] & 1) != 0:
                self.state[p] = self.state[p + 397 - 624] ^ (y >> 1) ^ 0x9908b0df
            else:
                self.state[p] = self.state[p + 397 - 624] ^ (y >> 1)
            p += 1
            
        y = (self.state[p] & 0x80000000) | (self.state[0] & 0x7fffffff)
        if (self.state[0] & 1) != 0:
             self.state[p] = self.state[p + 397 - 624] ^ (y >> 1) ^ 0x9908b0df
        else:
             self.state[p] = self.state[p + 397 - 624] ^ (y >> 1)

    # -------------------------------------------------------------------------
    # NOMBRE: genrand_int32
    # DESCRIPCIÓN: 
    #   Retorna un número entero aleatorio de 32 bits aplicando el proceso 
    #   de "tempering" para mejorar la equidistribución de los bits.
    # SALIDA: 
    #   - y [uint32]: Entero aleatorio de 32 bits.
    # -------------------------------------------------------------------------
    def genrand_int32(self):
        self.left -= 1
        if self.left == 0:
            self.next_state()
            
        y = self.state[self.inext]
        self.inext += 1

        y ^= (y >> 11)
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= (y >> 18)

        return y

    # -------------------------------------------------------------------------
    # NOMBRE: genrand_real3
    # DESCRIPCIÓN: 
    #   Genera un número real aleatorio en el intervalo abierto (0, 1).
    # SALIDA: 
    #   - [float64]: Valor real aleatorio.
    # -------------------------------------------------------------------------
    def genrand_real3(self):
        return (float(self.genrand_int32()) + 0.5) * (1.0/4294967296.0)

    # -------------------------------------------------------------------------
    # NOMBRE: genrand_real1
    # DESCRIPCIÓN: 
    #   Genera un número real aleatorio en el intervalo cerrado [0, 1].
    # SALIDA: 
    #   - [float64]: Valor real aleatorio.
    # -------------------------------------------------------------------------
    def genrand_real1(self):
        return float(self.genrand_int32()) * (1.0/4294967295.0)
    
    # -------------------------------------------------------------------------
    # NOMBRE: genrand_res53
    # DESCRIPCIÓN: 
    #   Genera un número real aleatorio con una resolución de 53 bits 
    #   en el intervalo [0, 1) utilizando dos extracciones de 32 bits.
    # SALIDA: 
    #   - [float64]: Valor real aleatorio de alta precisión.
    # -------------------------------------------------------------------------
    def genrand_res53(self):
        a = self.genrand_int32() >> 5
        b = self.genrand_int32() >> 6
        return (a * 67108864.0 + b) * (1.0/9007199254740992.0)

    # -------------------------------------------------------------------------
    # NOMBRE: genrand_gaussian
    # DESCRIPCIÓN: 
    #   Genera un valor con distribución normal aproximada basándose en 
    #   la suma de variables uniformes (lógica original del archivo).
    # SALIDA: 
    #   - a [float64]: Valor con distribución gaussiana aproximada.
    # -------------------------------------------------------------------------
    def genrand_gaussian(self):
        # Implementacion simple de Box-Muller usando los metodos propios
        # Ojo: la original sumaba 6 uniformes, NO es Box-Muller, es Teorema Limite Central aproximado
        # Reproducimos la logica original del archivo subido:
        # "a = 0.0; for i in range(6): a += genrand_real1(); a -= genrand_real1();"
        a = 0.0
        for i in range(6):
            a += self.genrand_real1()
            a -= self.genrand_real1()
        return a