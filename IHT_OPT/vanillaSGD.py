import torch
import numpy as np
from IHT_OPT.baseOptimizer import myOptimizer


###############################################################################################################################################################
# ---------------------------------------------------- VANILLA-SGD ------------------------------------------------------------------------------------------
###############################################################################################################################################################

class vanillaSGD(myOptimizer):
  def __init__(self,params,**kwargs):
    print(kwargs)
    super().__init__(params,**kwargs)

    # Internal States
    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['step'] = 0

    # Internal Variables
    self.methodName = "vanilla_SGD"

  # NOTE: y = p + 0.0
  # print(p.requires_grad) #<- True
  # print(y.requires_grad) #<- False
  @torch.no_grad()
  def step(self):
    #print(f"speed iteration {self.iteration}")
    #self.logging()

    #self.easyPrintParams()
    self.updateWeights()
    #self.easyPrintParams()
    self.iteration +=1

    #print('fixed vanilla SGD')
    return None

  # Regular Gradient Descent
  def updateWeights(self,**kwargs):
    #print("SGD updateWeights")
    # NOTE: unfortunately we do need the self keyword because we are using class instances
    for p in self.paramsIter():
        # NOTE TO FUTURE DIMITRI: you need to add an underscore else this is considered as an operation that returns something (I think)
        #p.add_(  (-1.0 / self.beta) * p.grad / pow(5, np.floor(self.iteration / 360))  )
        p.add_(  (-1.0 / self.beta) * p.grad )