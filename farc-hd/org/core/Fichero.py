# Created on 16-Jun-2004
# Clase implementada funciones para el manejo de ficheros de datos
# @author Jesus Alcalu Fernandez

# Java to python translator: Iñaki Mellado

import numpy as np
import sys

class Fichero:
    # -------------------------------------------------------------------------
    # Nombre funcion: leeFichero
    # Descripcion:
    #   Abre un archivo en modo lectura y devuelve todo su contenido en una sola cadena.
    # Explicaciones de las variables de entrada:
    #   - nombreFichero [str]: Ruta o nombre del archivo que se desea leer.
    # Explicacion de las variables de salida:
    #   - cadena [str]: Contenido completo del fichero.
    # -------------------------------------------------------------------------
    @staticmethod
    def leeFichero(nombreFichero):
        cadena = ""
        try:
            with open(nombreFichero, "r") as file:
                for line in file:
                    cadena += line
        except Exception as e:
            print(f"Error al leer el fichero {nombreFichero}: {e}")
            sys.exit()
        finally:
            return cadena
    
    # -------------------------------------------------------------------------
    # Nombre funcion: escribeFichero
    # Descripcion:
    #   Crea o sobreescribe un archivo con el contenido de la cadena proporcionada.
    # Explicaciones de las variables de entrada:
    #   - nombreFichero [str]: Ruta o nombre del archivo de destino.
    #   - cadena [str]: Texto que se desea volcar en el fichero.
    # Explicacion de las variables de salida:
    #   - None
    # -------------------------------------------------------------------------
    def escribeFichero(nombreFichero, cadena):
        try:
            with open(nombreFichero, "w") as file:
                file.write(cadena)
        except Exception as e:
            print(f"Error al escribir el fichero {nombreFichero}: {e}")
            sys.exit()
    
    # -------------------------------------------------------------------------
    # Nombre funcion: AnadirtoFichero
    # Descripcion:
    #   Abre un archivo en modo 'append' para añadir contenido al final del mismo 
    #   sin borrar lo existente.
    # Explicaciones de las variables de entrada:
    #   - nombreFichero [str]: Ruta o nombre del archivo.
    #   - cadena [str]: Texto que se desea añadir.
    # Explicacion de las variables de salida:
    #   - None
    # -------------------------------------------------------------------------
    def AnadirtoFichero (nombreFichero, cadena):
        try:
            with open(nombreFichero, "a") as file:
                file.write(cadena)
        except Exception as e:
            print(f"Error al escribir el fichero {nombreFichero}: {e}")
            sys.exit()