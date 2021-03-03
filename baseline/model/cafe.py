import utils
import time
import math
import numpy as np
import ast
import json


class Cafe():
    def __init__(self, X, k=50, s=500, teta=0.8):
        """
        parametros: 
            - X: conjunto dos termos candidatos
            - k: número de aspectos (default=50)
            - s: indica o número de candidatos que serão agrupados primeiro 
            para gerar os grupos de sementes (default=500)
            - teta: limite superior da distância entre dois agrupamentos
            (default=0.8)
        """
        self.X = X
        self.k = k
        self.s = s
        self.teta = teta
        
    def get_X(self):
        return self.X
    
    def get_k(self):
        return self.k
    
    def get_s(self):
        return self.s
    
    def get_teta(self):
        return self.teta
        