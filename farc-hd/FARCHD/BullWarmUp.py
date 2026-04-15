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

from farc_hd.FARCHD.Bull import Bull
# DESCRIPCIon:
#   Clase secundaria para precalentamiento de numba
class BullWarmUp(Bull):
    """
    Versión 'Muda' de Bull. 
    Ejecuta toda la lógica matemática (evolución, reglas, inferencia) 
    pero ANULA todas las funciones de escritura en disco (I/O).
    Esto evita que el calentamiento pierda tiempo creando archivos basura.
    """
    def writeEvo(self): pass
    def writeRules(self): pass
    def writeTime(self): pass
    def doOutput(self, dataset, filename): pass