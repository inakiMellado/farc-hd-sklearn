# ==============================================================================
# FARC-HD (Fuzzy Association Rule-based Classification Model for High-Dimensional)
# ==============================================================================
# Original parseParameters Java Implementation:
# @author Alberto Fernández
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

import sys
from farc_hd.org.core.Fichero import Fichero

class ParseParameters:
    def __init__(self):
        self.inputFiles = []
        self.outputFiles = []
        self.parameters = []
     
     
     # It obtains all the necesary information from the configuration file.<br/>
     # First of all it reads the name of the input data-sets, training, validation and test.<br/>
     # Then it reads the name of the output files, where the training (validation) and test outputs will be stored<br/>
     # Finally it read the parameters of the algorithm, such as the random seed.<br/>
     
     # @param fileName Name of the configuration file
    def parseConfigurationFile(self, fileName):
        file = Fichero.leeFichero(fileName)
        line = file.splitlines()

        self.readName(line)
        self.readInputFiles(line)
        self.readOutputFiles(line)
        self.readAllParameters(line)

        #self.printConfiguration()


    # It reads the name of the algorithm from the configuration file
    # @param line StringTokenizer It is the line containing the algorithm name.
    def readName(self, line):
        if not line:
            print ("Error: No algorithm name found in the configuration file")
            sys.exit()
        data = line.pop(0).split(" = ")
        data.pop(0)
        self.algorithmName = data.pop(0)
        for token in data:
            self.algorithmName += " " + token.strip()
        
    # We read the input data-set files and all the possible input files
    # @param line StringTokenizer It is the line containing the input files.
    def readInputFiles(self, line):
        if not line:
            print("Error: No input files found in the configuration file")
            sys.exit()
        line.pop(0)
        
        data = line.pop(0).split(" = ")
        data.pop(0)
        
        data = data[0].split(" ")
        self.trainingFile = data.pop(0).replace('"','')
        self.testFile = data.pop(0).replace('"','')
        self.validationFile = data.pop(0).replace('"','')
        for token in data:
            self.inputFiles.append(token.strip().replace('"',''))

    # We read the output files for training and test and all the possible remaining output files
    # @param line StringTokenizer It is the line containing the output files.

    def readOutputFiles(self, line):
        if not line:
            print("Error: No output files found in the configuration file")
            sys.exit()
        data = line.pop(0).split(" = ")
        data.pop(0)
        data = data[0].split(" ")
        self.outputTrFile = data.pop(0).replace('"','')
        self.outputTstFile = data.pop(0).replace('"','')
        for token in data:
            self.outputFiles.append(token.strip().replace('"',''))

    # We read all the possible parameters of the algorithm
    # @param line StringTokenizer It contains all the parameters.

    def readAllParameters(self, line):
        line.pop(0)
        while line:
            new_line = line.pop(0)
            data = new_line.split(" = ")
            self.parameters.append(data[1])
       ## If the algorithm is non-deterministic the first parameter is the Random SEED
    
    def getTrainingInputFile(self):
        return self.trainingFile

    def getTestInputFile(self):
        return self.testFile

    def getValidationInputFile(self):
        return self.validationFile

    def getTrainingOutputFile(self):
        return self.outputTrFile

    def getTestOutputFile(self):
        return self.outputTstFile

    def getAlgorithmName(self):
        return self.algorithmName

    def getParameters(self):
        return self.parameters

    def getParameter(self, pos):
        return self.parameters[pos]

    def getInputFiles(self):
        return self.inputFiles

    def getInputFile(self, pos):
        return self.inputFiles[pos]

    def getOutputFiles(self):
        return self.outputFiles

    def getOutputFile(self, pos):
        return self.outputFiles[pos]

    def printConfiguration(self):
        print("Algorithm: " + self.algorithmName)
        print("Training file: " + self.trainingFile)
        print("Test file: " + self.testFile)
        print("Validation file: " + self.validationFile)
        print("Output training file: " + self.outputTrFile)
        print("Output test file: " + self.outputTstFile)
        print("Parameters:")
        for i in range(len(self.parameters)):
            print(f"Parametro {i}: {self.parameters[i]}")
        print("-----------------------------------------------------")
