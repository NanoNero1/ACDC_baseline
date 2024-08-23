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

      state['zt'] = torch.zeros_like((p.to(self.device)))
      state['xt'] = p.data.detach().clone()
      state['zt_oldGrad'] = torch.zeros_like((p.to(self.device)))

    self.methodName = "vanilla_AGD"

  def step(self):
    #self.logging()
    self.updateWeights()
    self.iteration += 1

  ##############################################################################

  ### NOTE: this is not the actual function that is used in the end,
  # the real function is updateWeightsTwo
  def updateWeights(self):
    # Update z_t the according to the AGD equation in the note
    with torch.no_grad():
      for p in self.paramsIter():

        state = self.state[p]


        #First Get z_t+
        state['zt'] = (state['zt'] - (state['zt_oldGrad'] / self.beta) )

        # And then we do the actual update, NOTE: zt is actually z_t+ right now
        state['zt'] = (self.sqKappa / (self.sqKappa + 1.0) ) * state['zt'] + (1.0 / (self.sqKappa + 1.0)) * state['xt']

    self.getNewGrad('zt')

    with torch.no_grad():
      for p in self.paramsIter():
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
          p.data = state['xt'].clone().detach()

  def getNewGrad(self,iterate):
    with torch.no_grad():
      for p in self.paramsIter():
        state = self.state[p]

        p.data = state[iterate].clone().detach()

    self.zero_grad()
    data,target = self.currentDataBatch

    newOutput = self.model(data)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(newOutput,target)
    #loss = F.nll_loss(newOutput, target)
    loss.backward()

    if iterate == "zt":
      self.loss_zt = float(loss.clone().detach())

  def copyXT(self):
    with torch.no_grad():
      for p in self.paramsIter():
        state = self.state[p]
        state['xt'] = p.data.clone().detach()
