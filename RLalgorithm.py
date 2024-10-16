from abc import ABC

class RLAlgorithm(ABC):
    def __init__(self,gamma,alpha,maxEpisodes,epsilon):
        self.gamma = gamma
        self.alpha = alpha
        self.maxEpisodes = maxEpisodes
        self.epsilon = epsilon
        
