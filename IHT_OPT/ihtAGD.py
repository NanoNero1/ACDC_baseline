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

    self.specificSteps = 0

    

  def step(self):
    self.specificSteps += 1
    #self.trackingSparsity()
    #print(f"speed iteration {self.iteration}")

    # Sloppy but works
    #newSparsityIter = np.floor( (self.iteration - 100) / 80)
    #self.sparsity = min(0.9, 0.5 + 0.1*newSparsityIter)
    #self.easyPrintParams()
    #self.logging()

    #print('we got this far at least then')

    self.compressOrDecompress()
    #self.trackMatchingMasks(self)
    #self.iteration += 1

  #def returnSparse(self):

  def decompressed(self):
    self.areWeCompressed = False
    print('decompressed')
    self.updateWeightsTwo()

  def warmup(self):
    self.areWeCompressed = False
    print('warmup')
    self.updateWeightsTwo()

  # I checked this, it seems to work
  def truncateAndFreeze(self):
    
    self.updateWeightsTwo()
    self.areWeCompressed = True

    print('this should work')
    # define zt



    # Truncate xt
    self.sparsify()

    ## OFF
    #self.sparsify(iterate='zt')

    self.copyXT()


    # Freeze xt
    self.freeze()

    ## OFF
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
        #or  self.iteration >= self.startFineTune + 1):
        howFarAlong = ((self.iteration - self.warmupLength) % self.phaseLength) + 1
        # if self.areWeCompressed and (howFarAlong == 1):
        if self.iteration >= self.startFineTune:
          self.refreeze(iterate='zt')
        #   #pass


        # And then we do the actual update, NOTE: zt is actually z_t+ right now
        state['zt'] = (self.sqKappa / (self.sqKappa + 1.0) ) * state['zt'] + (1.0 / (self.sqKappa + 1.0)) * state['xt']

        #Find the new z_t
        #state['zt'] = (self.sqKappa / (self.sqKappa + 1.0) ) * (state['zt'] - (state['zt_oldGrad'] / self.beta) ) + (1.0 / (self.sqKappa + 1.0)) * state['xt']

    #self.sparsify(iterate='zt')

    # CAREFUL! this changes the parameters for the mode!
    self.getNewGrad('zt')
    #if self.specificSteps > 8000 and self.specificSteps < 10000:  
    #  self.clipGradients()
    ######### ALTERT ######## THERE SHOULD BE self.getNewGrad('zt') above!!!

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
    self.areWeCompressed = True
    print('compressed step')
    self.updateWeightsTwo()
    self.refreeze()

    ## OFF
    #self.refreeze('zt')

  def clipGradients(self,clipAmt=0.0001):
    print("I AM CLIPPING!!!!!!")

    #print(len(self.param_groups))
    #print("these are the groups")
    #abort()

    #torch.nn.utils.clip_grad_norm_(self.param_groups[0]['params'],norm_type='inf', max_norm=clipAmt)
    torch.nn.utils.clip_grad_value_(self.param_groups[0]['params'],clip_value=clipAmt)
    #for i 
    #torch.clamp(self.param_groups[0]['params'],min=-1.0*clipAmt,clipAmt=1.0)
    pass

  def trackMatchingMasks(self):
    concatMatchMask = torch.zeros((1)).to(self.device)
    for p in self.paramsIter():
      state = self.state[p]

      matchingMask = ((torch.abs(p.data) > 0).type(torch.uint8) == (torch.abs(state['zt'])).type(torch.uint8) > 0 ).type(torch.float)
      
      concatMatchMask = torch.cat((concatMatchMask,matchingMask),0)

    self.run[f"trials/{self.methodName}/matchingMasks"].append(torch.mean(matchingMask))


  def weightedSparsify(self,iterate):
    weightedWeights = torch.zeros((1)).to(self.device)
    with torch.no_grad():
      for p in self.paramsIter():
        if iterate == None:
          layer = p.data
        else:
          state = self.state[p]
          layer = state[iterate]

          weightedLayer = torch.flatten(torch.abs(layer) * torch.log(layer.size()))
          weightedWeights = torch.cat((weightedWeights,weightedLayer),0)
      
      topK = int(len(weightedWeights)*(1-self.sparsity))

      # All the top-k values are sorted in order, we take the last one as the cutoff
      vals, bestI = torch.topk(torch.abs(weightedWeights),topK,dim=0)
      weightedCutoff = vals[-1]

      for p in self.paramsIter():
        state = self.state[p]
        if iterate == None:
          #print("!!!!!!!!!!! this should sparsify the params")
          p.data[torch.abs(p) * torch.log(p.size()) <= weightedCutoff] = 0.0
        else:
          # NOTE: torch.abs(p) is wrong, maybe that's the bug
          (state[iterate])[torch.abs(state[iterate]) * torch.log(state[iterate].size()) <= weightedCutoff] = 0.0




    

    





  ##########################################