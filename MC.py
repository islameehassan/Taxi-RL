import numpy as np
import random
from utils import generate_random_walk_dataset
from RLalgorithm import RLAlgorithm
import matplotlib.pyplot as plt


class MC(RLAlgorithm):
    def __init__(self,S,A,gamma=0.9,alpha=0.3,max_episodes=100,epsilon=0.4):
        super().__init__(gamma,alpha,max_episodes,epsilon)
        self.S = S
        self.A = A
        self.V = {s: 0 for s in S}
        self.Q = {(s,a): 0 for s in S
                            for a in A}
    
    def predict(self,experience,rewards,isStateValue):
        """
            redirect to the appropriate value function based on the isStateValue and return
            the value function

            Args:
                experience (list): a list of all encountered states/actions during a specific episode
                rewards (list): a list of all rewards made
                isStateValue (bool): whether the evaluation is done on state value or state-action function
            
            Returns:
                value function(dict): either state value or state-action function
            
            Raises:
                Value error: if experience's length does not match rewards'
        """
        if len(experience) != len(rewards):
            raise ValueError("Experience list must have the same length as rewards list")
        
        if isStateValue:
            return self.__predict_state_value(experience,rewards)
        return self.__predict_state_action_value(experience,rewards)

    def __predict_state_value(self,states,rewards):
        G_t = 0      
        for index in range(len(states)-1,-1,-1):
            G_t = rewards[index] + self.gamma*G_t
            s = states[index]
            self.V[s] = self.V[s] + self.alpha*(G_t - self.V[s])
        
        return self.V

    def __predict_state_action_value(self,state_actions,rewards): 
        G_t = 0      
        for index in range(len(state_actions)-1,-1,-1):
            G_t = rewards[index] + self.gamma*G_t
            s,a = state_actions[index]
            self.Q[(s,a)] = self.Q[(s,a)] + self.alpha*(G_t - self.Q[(s,a)])
        
        return self.Q
    
    def select_action_epsilon_greedy(self,state):
        state_action_values = [self.Q[(state,i)] for i in self.A]
        best_action = np.argmax(state_action_values)
        actions_probs = [self.epsilon/len(self.A) for _ in self.A]
        actions_probs[best_action] += 1 - self.epsilon
        action_selected = random.choices(population=self.A,weights=actions_probs,k=1)[0]
        return action_selected



    def clear(self):
        self.V = {s: 0 for s in range(self.Sn)}
        self.Q = {(s,a): 0  for s in range(self.Sn)
                            for a in range(self.An)}
    

def main():
    statesSize = 5
    numEpisodes = 10000
    mc = MC(S = list(range(statesSize)),A=[-1,1])
    

    for _ in range(numEpisodes):
        experience,rewards = generate_random_walk_dataset(5,False)
        mc.predict(experience,rewards,False) 
    print(mc.Q)
    

main()
