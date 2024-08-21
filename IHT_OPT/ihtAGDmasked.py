import torch
from IHT_OPT.vanillaAGD import vanillaAGD
from IHT_OPT.ihtSGD import ihtSGD
import numpy as np

###############################################################################################################################################################
# ---------------------------------------------------- IHT-AGD ------------------------------------------------------------------------------------------
###############################################################################################################################################################

class ihtAGD(vanillaAGD,ihtSGD):
  def __init__(self,params,**kwargs):
    super().__init__(params,**kwargs)
    self.methodName = "iht_AGD"
    self.alpha = self.beta / self.kappa

  def step(self):
    self.trackingSparsity()
    #print(f"speed iteration {self.iteration}")

    # Sloppy but works
    #newSparsityIter = np.floor( (self.iteration - 100) / 80)
    #self.sparsity = min(0.9, 0.5 + 0.1*newSparsityIter)
    #self.easyPrintParams()
    #self.logging()

    #print('we got this far at least then')

    self.compressOrDecompress()
    self.iteration += 1

  #def returnSparse(self):

  def decompressed(self):
    print('decompressed')
    self.updateWeightsTwo()

  def warmup(self):
    print('warmup')
    self.updateWeightsTwo()

  # I checked this, it seems to work
  def truncateAndFreeze(self):
    self.updateWeightsTwo()
    print('this should work')
    # define zt



    # Truncate xt
    self.sparsify()
    self.copyXT()


    # Freeze xt
    self.freeze()

    # Freeze zt
    #self.freeze(iterate='zt')

    pass

  ##############################################################################

  def updateWeightsTwo(self):

    print("AGD updateWeights")
    # Update z_t the according to the AGD equation in the note

    with torch.no_grad():
      for p in self.paramsIter():

        state = self.state[p]


        #First Get z_t+
        state['zt'] = (state['zt'] - (state['zt_oldGrad'] / self.beta) )

        #Then sparsify z_t+
        ## NOTE to Dim: - you sparsify here
        self.sparsify(iterate='zt')

        # And then we do the actual update, NOTE: zt is actually z_t+ right now
        state['zt'] = (self.sqKappa / (self.sqKappa + 1.0) ) * state['zt'] + (1.0 / (self.sqKappa + 1.0)) * state['xt']

        #Find the new z_t
        #state['zt'] = (self.sqKappa / (self.sqKappa + 1.0) ) * (state['zt'] - (state['zt_oldGrad'] / self.beta) ) + (1.0 / (self.sqKappa + 1.0)) * state['xt']

    #self.sparsify(iterate='zt')

    # CAREFUL! this changes the parameters for the mode!

    with torch.no_grad():
      for p in self.paramsIter():
        #print(p.grad)
        # CHECK: Is it still the same state?
        state = self.state[p]
        state['zt_oldGrad'] = p.grad.clone().detach()

        # NOTE: p.grad is now the gradient at zt
        p.data = state['xt'] - (1.0 / pow(self.alpha*self.beta , 0.5)) * p.grad

    # We need to keep a separate storage of xt because we replace the actual network parameters
    self.copyXT()


  def compressedStep(self):
    print('compressed step')
    self.updateWeightsTwo()
    self.refreeze()
    #self.refreeze('zt')

  def maskedSparsify(self,iterate='xt'):
    for p in self.paramsIter():
      state = self.state[p]
      oldWeight = state[f"{iterate}_oldWeight"]
      diffWeights = p - oldWeight
      ## and then the old cutoff?
      ## or concat to a big file
      




  ##########################################