import numpy as np
from collections import OrderedDict
import re

class myDataSet:
    def __init__(self):
        self.data = np.array([])
        self.target = np.array([])
        self.feature_names = []
        self.target_names = []
        self.hasMissingValues = False
        self.nData = 0
        self.nClasses = 0
        self.nInputs = 0
        self.infoAtributos = OrderedDict()
        self.instancesCl = np.array([])
        self.frecuentCl = np.array([])

        self.categorical = []
        self.integer = []
    
    # -------------------------------------------------------------------------
    # NOMBRE: set_data_from_numpy
    # DESCRIPCIÓN: 
    #   Carga datos desde matrices Numpy.
    # ENTRADA:
    #   - X [np.array]: Matriz de inputs
    #   - y [np.array]: Matriz de la clase
    # -------------------------------------------------------------------------
    def set_data_from_numpy(self, X, y, categorical_flag, categorical_variables):
        self.data = X
        self.target = y
        self.nData = X.shape[0]
        self.nInputs = X.shape[1]
        
        # Detectar clases únicas
        unique_classes = np.unique(y)
        self.nClasses = len(unique_classes)
        
        # Generar nombres genéricos
        self.feature_names = [f"Var_{i}" for i in range(self.nInputs)]
        self.target_names = [f"Class_{i}" for i in range(self.nClasses)]
        
        # Valores faltantes
        self.hasMissingValues = np.isnan(X).any()
        
        # Calcular rangos y tipo de dato
        #necesario para calculo de tipos (por si usuario introduce un [2] o [False, True, False])
        cats_explicitas = set()
        if categorical_flag and categorical_variables is not None:
            if isinstance(categorical_variables[0], bool):
                cats_explicitas = {i for i, val in enumerate(categorical_variables) if val}
            else:
                cats_explicitas = set(categorical_variables)

        for i in range(self.nInputs):
            col = X[:, i]
            col_limpia = col[np.isfinite(col)]
            #################################################################
            if len(col_limpia) == 0:
                self.categorical.append(False)
                self.integer.append(False)
                continue
                
            hay_reales = np.any(col_limpia % 1 != 0)

            
            if categorical_flag and (i in cats_explicitas):
                # Se sabe que es categórica y esta en la lista == Caregorica
                self.categorical.append(True)
                self.integer.append(False)
                
            elif categorical_flag and (i not in cats_explicitas):
                # Se sabe que es categórica y no esta en la lista == Entera/Real
                self.categorical.append(False)
                if hay_reales:
                    self.integer.append(False)
                else:
                    self.integer.append(True)
                    
            else:
                # No se sabe cuales son categoricas
                if hay_reales:
                    # Ni entero ni categorico
                    self.categorical.append(False)
                    self.integer.append(False)
                else:
                    valores_unicos = np.unique(col_limpia)
                    tope_maximo_categorias = 10
                    
                    if len(valores_unicos) <= tope_maximo_categorias:
                        serie_ideal = np.arange(len(valores_unicos))
                        es_serie_0_a_N = np.array_equal(valores_unicos, serie_ideal)
                        
                        if es_serie_0_a_N:
                            # Asumimos Categorica
                            self.categorical.append(True)
                            self.integer.append(False)
                        else:
                            self.categorical.append(False)
                            self.integer.append(True)
                    else:
                        # Asumimos entero
                        self.categorical.append(False)
                        self.integer.append(True)


            #################################################################
            # Caluclos de rangos
            if self.categorical[-1] == True:
                # Si es categórica, almacena los valores únicos en forma de lista
                valores_unicos = np.unique(col_limpia)
                self.infoAtributos[self.feature_names[i]] = valores_unicos.tolist()
            else:
                # Si es continua o entera, almacena [min, max]
                min_val = np.nanmin(col_limpia)
                max_val = np.nanmax(col_limpia)
                
                # Evitar rango 0 si todos los valores son iguales
                if min_val == max_val:
                    min_val -= 0.001
                    max_val += 0.001
                self.infoAtributos[self.feature_names[i]] = [min_val, max_val]
            #################################################################
            # Calcular frecuencias de clase (necesario para el cálculo de WRACC/Apriori)
            self.computeInstancesPerClass()


    def lecturaDatos(self, fileE, isTrain):
        self.isTrain = isTrain
        try:
            f=open(fileE,'r')
            line=f.readline()
            att=OrderedDict()
            self.data = []
            self.target = []
            self.feature_names = []
            self.target_names = []
            self.categorical = []
            self.integer = []
            while line.find("@data")<0:
                if "@attribute" in line and ("real" in line or "integer" in line):
                    # Extraer el nombre del atributo
                    nombre = line.split()[1]

                    # Extraer el contenido entre corchetes [min, max]
                    rango = line[line.find("[") + 1: line.find("]")]

                    # Separar por coma y convertir a float
                    minimo, maximo = [float(x.strip()) for x in rango.split(",")]

                    # Guardar en el diccionario
                    attAux = {nombre: [minimo, maximo]}
                    att.update(attAux)
                    self.categorical.append(False)
                    self.integer.append("integer" in line)
                elif "@attribute" in line and ("real" not in line) and ("integer" not in line):
                    # nombre del atributo
                    name = line.split()[1]
                    # extraer contenido entre { }
                    content = line[line.find("{") + 1: line.find("}")]
                    # separar valores, quitando espacios
                    values = [v.strip() for v in content.split(",")]
                    attAux = {name: values}
                    att.update(attAux)
                    self.categorical.append(True)
                    self.integer.append(False)
                elif line.find("@output")>=0 or line.find("@outputs")>=0:
                    clase = line.split()
                    clase = clase[1]
                line=f.readline()
            auxClases = att.pop(clase)
            self.categorical = self.categorical[:-1]
            clases = auxClases[:]
            attAux={clase:clases}
            att.update(attAux)
            line=f.readline()
            exClases = []
            examples = []
            exOriginal = []
            while line!="":
                line=line.replace(","," ")
                l=line.split()
                values = l[0:len(l)-1]
                val = []
                valOriginal = []
                for i, v in enumerate(values):
                    if self.categorical[i]:
                        listaClaves = list(att)
                        lista = att[listaClaves[i]]
                        val.append(lista.index(v))
                        valOriginal.append(v)
                    else:
                        val.append(float(v))
                        valOriginal.append(float(v))
                examples.append(val)
                exOriginal.append(valOriginal)
                lista = sorted(att[clase])
                exClases.append(lista.index(l[len(l) - 1]))
                line=f.readline()
            examples = np.array(examples)
            f.close()
            self.data = examples
            self.target = np.array(exClases)
            aux = list(att)
            self.feature_names = aux[:-1]
            self.target_names = att[clase]
            self.infoAtributos = att

            self.nData = self.data.shape[0]
            self.nInputs = self.data.shape[1]
            # Detectar clases únicas
            unique_classes = np.unique(self.target)
            self.nClasses = len(unique_classes)
            # Valores faltantes
            self.hasMissingValues = np.isnan(self.data).any()

        except Exception as ex:
            print(f"Error en la lectura de datos: {ex}")
            if 'f' in locals() and not f.closed:
                f.close()
        finally:
            if hasattr(self, 'nData') and self.nData > 0:
                self.computeInstancesPerClass()
            else:
                print("    > No instances read or error occurred.")
    # -------------------------------------------------------------------------
    # NOMBRE: isMissingValues
    # DESCRIPCIÓN: 
    #   Indica si el dataset cargado contiene valores perdidos.
    # SALIDA: 
    #   - [bool]: True si existen valores perdidos.
    # -------------------------------------------------------------------------
    def isMissingValues(self):
        return self.hasMissingValues

    # -------------------------------------------------------------------------
    # NOMBRE: returnRanks
    # DESCRIPCIÓN: 
    #   Devuelve los rangos (mínimos y máximos) de todas las variables de entrada.
    # SALIDA: 
    #   - rangos [list]: Lista de pares [mín, máx] por variable.
    # -------------------------------------------------------------------------
    def returnRanks(self):
        rangos = []
        # Iteramos estrictamente por las variables de entrada reales
        for i in range(self.getnInputs()):
            nombre = self.feature_names[i]
            valor = self.infoAtributos[nombre]
            if self.isNominal(i):
                # Si es nominal, el rango es [0, número de categorías - 1]
                rangos.append([0.0, float(len(valor) - 1)])
            else:
                # Si es numérico, guardamos el [min, max]
                rangos.append(valor)
                
        # Añadimos el rango ficticio [0.0, 0.0] para la salida (exigencia del algoritmo original)
        rangos.append([0.0, 0.0])
        return rangos

    # -------------------------------------------------------------------------
    # NOMBRE: getnInputs
    # DESCRIPCIÓN: 
    #   Retorna el número de variables de entrada.
    # SALIDA: 
    #   - [int]: Cantidad de atributos de entrada.
    # -------------------------------------------------------------------------
    def getnInputs(self):
        return len(self.feature_names)

    # -------------------------------------------------------------------------
    # NOMBRE: names
    # DESCRIPCIÓN: 
    #   Retorna los nombres de los atributos de entrada.
    # SALIDA: 
    #   - [list]: Lista de nombres de las variables.
    # -------------------------------------------------------------------------
    def names(self):
        return self.feature_names

    # -------------------------------------------------------------------------
    # NOMBRE: clases
    # DESCRIPCIÓN: 
    #   Retorna los nombres de las clases de salida.
    # SALIDA: 
    #   - [list]: Lista de nombres de las clases.
    # -------------------------------------------------------------------------
    def clases(self):
        return self.target_names
    
        
    # -------------------------------------------------------------------------
    # NOMBRE: getnClasses
    # DESCRIPCIÓN: 
    #   Retorna el número total de clases en el dataset.
    # SALIDA: 
    #   - [int]: Cantidad de clases.
    # -------------------------------------------------------------------------
    def getnClasses(self):
        return self.nClasses
    
    # -------------------------------------------------------------------------
    # NOMBRE: getnData
    # DESCRIPCIÓN: 
    #   Retorna el número total de registros/ejemplos.
    # SALIDA: 
    #   - [int]: Cantidad de ejemplos.
    # -------------------------------------------------------------------------
    def getnData(self):
        return self.nData
    
    # -------------------------------------------------------------------------
    # NOMBRE: getnVars
    # DESCRIPCIÓN: 
    #   Retorna el número de variables de entrada.
    # SALIDA: 
    #   - [int]: Cantidad de atributos.
    # -------------------------------------------------------------------------
    def getnVars (self):
        return len(self.feature_names)

    # -------------------------------------------------------------------------
    # NOMBRE: getExample
    # DESCRIPCIÓN: 
    #   Recupera un ejemplo concreto del dataset.
    # ENTRADA:
    #   - pos [int]: Índice del ejemplo.
    # SALIDA: 
    #   - [np.array]: Vector de valores del ejemplo.
    # -------------------------------------------------------------------------
    def getExample (self, pos):
        return self.data[pos]
    
    # -------------------------------------------------------------------------
    # NOMBRE: getDataVariable
    # DESCRIPCIÓN: 
    #   Recupera todos los valores de una variable específica.
    # ENTRADA:
    #   - pos [int]: Índice de la variable.
    # SALIDA: 
    #   - [np.array]: Columna de datos de la variable.
    # -------------------------------------------------------------------------
    def getDataVariable(self, pos):
        return self.data[:, pos]
    
    # -------------------------------------------------------------------------
    # NOMBRE: getAllData
    # DESCRIPCIÓN: 
    #   Retorna la matriz completa de datos de entrada.
    # SALIDA: 
    #   - [np.array]: Matriz de datos.
    # -------------------------------------------------------------------------
    def getAllData (self):
        return self.data
    
    # -------------------------------------------------------------------------
    # NOMBRE: getOutputAsInteger
    # DESCRIPCIÓN: 
    #   Retorna la clase de un ejemplo específico como entero.
    # ENTRADA:
    #   - pos [int]: Índice del ejemplo.
    # SALIDA: 
    #   - [int]: Índice de la clase.
    # -------------------------------------------------------------------------
    def getOutputAsInteger(self, pos):
        return self.target[pos]
    
    # -------------------------------------------------------------------------
    # NOMBRE: getOutputsAsIntegers
    # DESCRIPCIÓN: 
    #   Retorna el vector completo de clases como enteros.
    # SALIDA: 
    #   - [np.array]: Vector de clases.
    # -------------------------------------------------------------------------
    def getOutputsAsIntegers(self):
        return self.target
    
    # -------------------------------------------------------------------------
    # NOMBRE: getOutputAsString
    # DESCRIPCIÓN: 
    #   Retorna el nombre de la clase de un ejemplo específico.
    # ENTRADA:
    #   - pos [int]: Índice del ejemplo.
    # SALIDA: 
    #   - [str]: Etiqueta textual de la clase.
    # -------------------------------------------------------------------------
    def getOutputAsString(self, pos):
        return self.target_names[self.target[pos]]

    # -------------------------------------------------------------------------
    # NOMBRE: isNominal
    # DESCRIPCIÓN: 
    #   Determina si una variable dada es categórica/nominal.
    # ENTRADA:
    #   - i [int]: Índice de la variable.
    # SALIDA: 
    #   - [bool]: True si es nominal.
    # -------------------------------------------------------------------------
    def isNominal(self, i):
        if i < len(self.categorical):
            return self.categorical[i]
        return False

    # -------------------------------------------------------------------------
    # NOMBRE: isInteger
    # DESCRIPCIÓN: 
    #   Determina si una variable dada es de tipo entero.
    # ENTRADA:
    #   - i [int]: Índice de la variable.
    # SALIDA: 
    #   - [bool]: True si es de tipo entero.
    # -------------------------------------------------------------------------
    def isInteger(self, i):
        if i < len(self.integer):
            return self.integer[i]
        return False

    # -------------------------------------------------------------------------
    # NOMBRE: computeInstancesPerClass
    # DESCRIPCIÓN: 
    #   Calcula el número de instancias y la frecuencia por cada clase.
    # -------------------------------------------------------------------------
    def computeInstancesPerClass(self):
        self.instancesCl = np.zeros(self.nClasses)
        self.frecuentCl = np.zeros(self.nClasses)

        for i in range(self.getnData()):
            self.instancesCl[self.target[i]] +=1

        for i in range(self.nClasses):
            self.frecuentCl[i] = (1.0 * self.instancesCl[i]) / float(self.nData)
    # -------------------------------------------------------------------------
    # NOMBRE: frecuentClass
    # DESCRIPCIÓN: 
    #   Retorna la frecuencia relativa de una clase específica.
    # ENTRADA:
    #   - clas [int]: Índice de la clase.
    # SALIDA: 
    #   - [float]: Frecuencia de la clase.
    # -------------------------------------------------------------------------
    def frecuentClass(self, clas):
        return self.frecuentCl[clas]

    # -------------------------------------------------------------------------
    # NOMBRE: numberInstances
    # DESCRIPCIÓN: 
    #   Retorna el número absoluto de instancias de una clase.
    # ENTRADA:
    #   - clas [int]: Índice de la clase.
    # SALIDA: 
    #   - [float]: Cantidad de instancias.
    # -------------------------------------------------------------------------
    def numberInstances(self, clas):
        return self.instancesCl[clas]

    # -------------------------------------------------------------------------
    # NOMBRE: copyHeader
    # DESCRIPCIÓN: 
    #   Genera una cadena con la cabecera completa del formato de datos.
    # SALIDA: 
    #   - p [str]: Cabecera formateada.
    # -------------------------------------------------------------------------
    def copyHeader (self):
        p = ""
        p = "@relation " + self.getRelationName() + "\n"
        p += self.getInputAttributesHeader()
        p += self.getOutputAttributesHeader()
        p += self.getInputHeader() + "\n"
        p += self.getOutputHeader() + "\n"
        p += "@data\n"
        return p
    
    # -------------------------------------------------------------------------
    # NOMBRE: getRelationName
    # DESCRIPCIÓN: 
    #   Retorna el nombre de la relación definida en el archivo.
    # SALIDA: 
    #   - r [str]: Nombre de la relación.
    # -------------------------------------------------------------------------
    def getRelationName (self):
        r = ""
        if hasattr(self, 'relation'):
            for i in range (len(self.relation)):
                r += self.relation[i]
        return r

    # -------------------------------------------------------------------------
    # NOMBRE: getInputAttributesHeader
    # DESCRIPCIÓN: 
    #   Genera las líneas @attribute para los atributos de entrada.
    # SALIDA: 
    #   - header [str]: Fragmento de cabecera de atributos.
    # -------------------------------------------------------------------------
    def getInputAttributesHeader(self):
        header = ""
        for i, name in enumerate(self.feature_names):
            min_val, max_val = self.infoAtributos[name]
            header += f"@attribute {name} real [{min_val}, {max_val}]\n"
        return header

    # -------------------------------------------------------------------------
    # NOMBRE: getOutputAttributesHeader
    # DESCRIPCIÓN: 
    #   Genera la línea @attribute para la variable de salida.
    # SALIDA: 
    #   - [str]: Línea de atributo de salida.
    # -------------------------------------------------------------------------
    def getOutputAttributesHeader(self):
        valores = ", ".join(self.target_names)
        return f"@attribute Class {{{valores}}}\n"

    # -------------------------------------------------------------------------
    # NOMBRE: getInputHeader
    # DESCRIPCIÓN: 
    #   Genera la línea @inputs con los nombres de las variables.
    # SALIDA: 
    #   - [str]: Línea de entradas.
    # -------------------------------------------------------------------------
    def getInputHeader(self):
        return f"@inputs {', '.join(self.feature_names)}"

    # -------------------------------------------------------------------------
    # NOMBRE: getOutputHeader
    # DESCRIPCIÓN: 
    #   Genera la línea @outputs con el nombre de la variable objetivo.
    # SALIDA: 
    #   - [str]: Línea de salidas.
    # -------------------------------------------------------------------------
    def getOutputHeader(self):
        return "@outputs Class"

    # -------------------------------------------------------------------------
    # NOMBRE: getOutputValue
    # DESCRIPCIÓN: 
    #   Convierte un índice de clase en su valor textual.
    # ENTRADA:
    #   - intValue [int]: Índice de la clase.
    # SALIDA: 
    #   - [str]: Nombre de la clase.
    # -------------------------------------------------------------------------
    def getOutputValue(self, intValue):
        return self.target_names[intValue]