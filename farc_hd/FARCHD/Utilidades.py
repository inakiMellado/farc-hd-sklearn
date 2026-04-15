import sys
import os

class DualLogger(object):
    # -------------------------------------------------------------------------
    # NOMBRE: __init__
    # DESCRIPCIÓN: 
    #   Constructor de DualLogger. Configura la duplicación de la salida 
    #   estándar para que el texto se envíe simultáneamente a la terminal y 
    #   a un archivo físico.
    # ENTRADA:
    #   - filename [str]: Ruta y nombre del archivo donde se guardará el log.
    # -------------------------------------------------------------------------
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    # -------------------------------------------------------------------------
    # NOMBRE: write
    # DESCRIPCIÓN: 
    #   Escribe un mensaje de forma síncrona en ambos flujos de salida 
    #   (consola y fichero).
    # ENTRADA:
    #   - message [str]: Cadena de texto a imprimir/guardar.
    # -------------------------------------------------------------------------
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    # -------------------------------------------------------------------------
    # NOMBRE: flush
    # DESCRIPCIÓN: 
    #   Asegura que todos los datos en los buffers de salida se escriban 
    #   físicamente en sus destinos.
    # -------------------------------------------------------------------------
    def flush(self):
        self.terminal.flush()
        self.log.flush()

    # -------------------------------------------------------------------------
    # NOMBRE: close
    # DESCRIPCIÓN: 
    #   Cierra formalmente el descriptor del archivo de log.
    # -------------------------------------------------------------------------
    def close(self):
        self.log.close()

# DATOS HARDCODEADOS PARA CALENTAMIENTO
CONTENT = """@relation unknow
@attribute A real [4.3, 7.9]
@attribute B real [2.0, 4.4]
@attribute C real [1.0, 6.9]
@attribute class {D}
@inputs A, B, C
@outputs class
@data
5.1, 3.5, 1.4, D
"""
# -------------------------------------------------------------------------
# NOMBRE: generate_robust_dummy
# DESCRIPCIÓN: 
#   Crea un archivo de dataset ARFF mínimo y estático. Se utiliza 
#   principalmente para el "calentamiento" (warm-up) de las funciones JIT 
#   de Numba, asegurando que la primera ejecución real sea rápida.
# ENTRADA:
#   - filename [str]: Nombre del archivo dummy a generar.
# -------------------------------------------------------------------------
def generate_robust_dummy(filename):
    """
    Genera el dataset IRIS estático para el calentamiento JIT (Numba).
    Se ignoran n_rows y n_vars, pero se mantienen como argumentos 
    para compatibilidad con llamadas existentes.
    """
    with open(filename, "w", encoding='utf-8') as f:
        f.write(CONTENT)

class HiddenPrints:
    # -------------------------------------------------------------------------
    # NOMBRE: __enter__
    # DESCRIPCIÓN: 
    #   Método de entrada del gestor de contexto. Redirige la salida estándar 
    #   (stdout) a os.devnull para silenciar cualquier impresión en consola.
    # -------------------------------------------------------------------------
    def __enter__(self):
        self._original_stdout = sys.stdout
        self.devnull = open(os.devnull, 'w')
        sys.stdout = self.devnull

    # -------------------------------------------------------------------------
    # NOMBRE: __exit__
    # DESCRIPCIÓN: 
    #   Método de salida del gestor de contexto. Restaura la salida estándar 
    #   original y cierra el puntero a devnull.
    # ENTRADA:
    #   - exc_type, exc_val, exc_tb: Información sobre excepciones si ocurrieron.
    # -------------------------------------------------------------------------
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        self.devnull.close()