from farc_hd.FARCHD.parseParameters import ParseParameters

class WarmUpParameters(ParseParameters):
    """
    Parámetros forzados para el calentamiento.
    Permite inyectar dinámicamente el tipo de inferencia (0 o 1).
    """
    def __init__(self, dummy_filename):
        super().__init__()
        self.dummy_file = dummy_filename
        self.forced_inference = "0"  # Se cambiará dinámicamente

    def getTrainingInputFile(self): return self.dummy_file
    def getValidationInputFile(self): return self.dummy_file
    def getTestInputFile(self): return self.dummy_file
    def getAlgorithmName(self): return "JIT-WarmUp"
    def getTrainingOutputFile(self): return "trash.tra"
    def getTestOutputFile(self): return "trash.tst"
    def getOutputFile(self, idx): return f"trash_{idx}.txt"

    def getParameter(self, pos):
        # Configuración estándar para activar todos los kernels de Numba
        # 0: Seed, 1: nLabels, 2: minsup, 3: minconf, 4: depth, 
        # 5: K, 6: maxTrials, 7: popSize, 8: alpha, 9: BITS_GEN, 10: typeInference
        params = [
            "53743421", "5", "0.5", "0.5", "1", "2", "20000", "5", "0.02", "30", 
            self.forced_inference # <--- Inyección dinámica
        ]
        if pos < len(params): return params[pos]
        return "0"