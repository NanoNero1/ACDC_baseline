import torch
from IHT_OPT.vanillaSGD import vanillaSGD
import torch.nn.functional as F


###############################################################################################################################################################
# ---------------------------------------------------- VANILLA-AGD ------------------------------------------------------------------------------------------
###############################################################################################################################################################

class vanillaAGD(vanillaSGD):
  def __init__(self,params,**kwargs):
    super().__init__(params,**kwargs)

    # Objective Function Property Variables
    self.alpha = self.beta / self.kappa
    self.sqKappa = pow(self.kappa,0.5)
    self.loss_zt = 0.0


    for p in self.paramsIter():
      state = self.state[p]

      # CHECK: is is ok to access it like this?
      state['zt'] = torch.zeros_like((p.to(self.device)))
      state['xt'] = p.data.detach().clone()
      state['zt_oldGrad'] = torch.zeros_like((p.to(self.device)))

    self.methodName = "vanilla_AGD"

  # NOTE: we want to turn this off?
  #@torch.no_grad - not sure if the activate it here since we calculate the gradient as nested in this function
  def step(self):
    #print("This is the fixed Accelerated Gradient Descent")
    #print(f"speed iteration {self.iteration}")
    #self.logging()
    self.updateWeights()
    self.iteration += 1

  ##############################################################################

  def updateWeights(self):
    #print("AGD updateWeights")
    # Update z_t the according to the AGD equation in the note
    with torch.no_grad():
      for p in self.paramsIter():

        state = self.state[p]


        #First Get z_t+
        state['zt'] = (state['zt'] - (state['zt_oldGrad'] / self.beta) )

        # And then we do the actual update, NOTE: zt is actually z_t+ right now
        state['zt'] = (self.sqKappa / (self.sqKappa + 1.0) ) * state['zt'] + (1.0 / (self.sqKappa + 1.0)) * state['xt']

        #Find the new z_t
        #state['zt'] = (self.sqKappa / (self.sqKappa + 1.0) ) * (state['zt'] - (state['zt_oldGrad'] / self.beta) ) + (1.0 / (self.sqKappa + 1.0)) * state['xt']

    # CAREFUL! this changes the parameters for the model
    self.getNewGrad('zt')

    with torch.no_grad():
      for p in self.paramsIter():
        # CHECK: Is it still the same state?
        state = self.state[p]
        state['zt_oldGrad'] = p.grad.clone().detach()

        # NOTE: p.grad is now the gradient at zt
        p.data = state['xt'] - (1.0 / pow(self.alpha*self.beta , 0.5)) * p.grad

    # We need to keep a separate storage of xt because we replace the actual network parameters
    self.copyXT()

  ##########################################

  def reCopyXt(self):
    with torch.no_grad():
        for p in self.paramsIter():
          state = self.state[p]

          # CHECK: is the order of operations correct?
          p.data = state['xt'].clone().detach()

  def getNewGrad(self,iterate):
    with torch.no_grad():
      for p in self.paramsIter():
        state = self.state[p]

        # CHECK: is the order of operations correct?
        p.data = state[iterate].clone().detach()
    #print('FIXED IHT-AGD')

    self.zero_grad()
    data,target = self.currentDataBatch

    newOutput = self.model(data)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(newOutput,target)
    #loss = F.nll_loss(newOutput, target)
    loss.backward()

    if iterate == "zt":
      self.loss_zt = float(loss.clone().detach())

    

    # CHECK: see if this works as intended
    #for p in self.paramsIter():
      #print(p)
      #print('and now the gradient')
      #print(p.grad)

  def copyXT(self):
    with torch.no_grad():
      for p in self.paramsIter():
        state = self.state[p]
        state['xt'] = p.data.clone().detach()
