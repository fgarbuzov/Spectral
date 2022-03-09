from .tensorfield import *
import pickle 

class Simulation:
    def __init__(self, time, sln, sln_tder, energy):
        self.time = time
        self.sln = sln
        self.sln_tder = sln_tder
        self.enrg = energy
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        return "Saved to " + filename
    
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)