# @author Written by Jesus Alcala (University of Granada) 16/06/2004
# @author Modified by Pedro Gonzalez (University of Jaen) 22/12/2008
# @version 2
# @since JDK1.5

class Files:
    # Implements methods to manage data files

    # -------------------------------------------------------------------------
    # Nombre funcion: readFile
    # Descripcion:
    #   Abre un archivo en modo lectura binaria, lo lee y decodifica su 
    #   contenido a una cadena de texto UTF-8.
    # Explicaciones de las variables de entrada:
    #   - file_name [str]: Ruta o nombre del archivo que se desea leer.
    # Explicacion de las variables de salida:
    #   - content [str]: Cadena con el contenido completo del archivo.
    # -------------------------------------------------------------------------
    def readFile(file_name):
        content = ""
        try:
            with open(file_name, "rb") as f:
                content = f.read().decode("utf-8")
        except IOError as e:
            print(f"Error reading file: {e}")
            exit(-1)
        return content

    # -------------------------------------------------------------------------
    # Nombre funcion: writeFile
    # Descripcion:
    #   Crea o sobrescribe un archivo con el contenido de texto proporcionado 
    #   utilizando codificación UTF-8.
    # Explicaciones de las variables de entrada:
    #   - file_name [str]: Ruta o nombre del archivo de destino.
    #   - content [str]: Texto que se desea escribir en el archivo.
    # Explicacion de las variables de salida:
    #   - None
    # -------------------------------------------------------------------------
    def writeFile(file_name, content):
        try:
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(content)
        except IOError as e:
            print(f"Error writing to file: {e}")
            exit(-1)

    # -------------------------------------------------------------------------
    # Nombre funcion: addToFile
    # Descripcion:
    #   Añade contenido al final de un archivo existente sin borrar su contenido 
    #   previo (modo append) usando codificación UTF-8.
    # Explicaciones de las variables de entrada:
    #   - file_name [str]: Ruta o nombre del archivo.
    #   - content [str]: Texto que se desea anexar al final.
    # Explicacion de las variables de salida:
    #   - None
    # -------------------------------------------------------------------------
    def addToFile(file_name, content):
        try:
            with open(file_name, "a", encoding="utf-8") as f:
                f.write(content)
        except IOError as e:
            print(f"Error appending to file: {e}")
            exit(-1)