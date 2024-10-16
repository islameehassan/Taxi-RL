from RLalgorithm import RLAlgorithm
import numpy as np
import random

class TD(RLAlgorithm):
    def __init__(self,S,A,gamma=0.9,alpha=0.3,max_episodes=100,epsilon=0.2):
        super().__init__(gamma,alpha,max_episodes,epsilon)
        self.S = S
        self.A = A
        self.V = {s: 0 for s in S}
        self.Q = {(s,a): 0 for s in S
                            for a in A}
    
    def select_action_epsilon_greedy(self,state):
        state_action_values = [self.Q[(state,i)] for i in self.A]
        best_action = np.argmax(state_action_values)
        actions_probs = [self.epsilon/len(self.A) for _ in self.A]
        actions_probs[best_action] += 1 - self.epsilon
        action_selected = random.choices(population=self.A,weights=actions_probs,k=1)[0]
        return action_selected 
    
    def SARSA(self,state,action,next_state,reward):
        next_action = self.select_action_epsilon_greedy(next_state)
        self.Q[(state,action)] += self.alpha*(reward + self.gamma*self.Q[(next_state,next_action)] - self.Q[(state,action)])

    
    def Q_learning(self,state,action,next_state,reward):
        best_action = np.argmax([self.Q[next_state,i] for i in self.A])
        self.Q[(state,action)] += self.alpha*(reward + self.gamma*self.Q[(next_state,best_action)] - self.Q[(state,action)])