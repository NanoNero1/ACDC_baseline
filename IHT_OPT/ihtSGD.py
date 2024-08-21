import torch
from IHT_OPT.vanillaSGD import vanillaSGD
import numpy as np

###############################################################################################################################################################
# ---------------------------------------------------- IHT-SGD ------------------------------------------------------------------------------------------
###############################################################################################################################################################
class ihtSGD(vanillaSGD):
  def __init__(self, params, sparsifyInterval=10,**kwargs):

    super().__init__(params,**kwargs)
    self.sparsifyInterval = sparsifyInterval

    # Compression, Decompression and Freezing Variables

    ## CIFAR10
    # self.phaseLength = 10
    # self.compressionRatio = 0.5
    # self.freezingRatio = 0.2
    # self.warmupLength = 6
    # self.startFineTune = 50

    ## MNIST
    # self.phaseLength = 4
    # self.compressionRatio = 0.5
    # self.freezingRatio = 0.2
    # self.warmupLength = 1
    # self.startFineTune = 16

    ## CIFAR100
    self.phaseLength = 20
    self.compressionRatio = 0.5
    self.freezingRatio = 0.2
    self.warmupLength = 10
    self.startFineTune = 170

    self.areWeCompressed = False
    self.notFrozenYet = True

    self.batchIndex = 0

    # State Initialization
    for p in self.paramsIter():
      state = self.state[p]
      state['xt_frozen'] = torch.ones_like(p)
      state['xt_gradient'] = torch.zeros_like(p)

    self.methodName = "iht_SGD"

    # Sparsity Tracking
    self.trackSparsity = 0
    self.trackSparsityLinear = 0
    self.trackSparsityBias = 0

  @torch.no_grad()
  def step(self):
    print('FIXED IHT SGD')
    print(f"speed iteration {self.iteration}")


    # Sloppy but works
    #newSparsityIter = np.floor( (self.iteration - 100) / 80)
    #self.sparsity = min(0.9, 0.5 + 0.1*newSparsityIter)

    #self.logging()
    #self.easyPrintParams()
    self.compressOrDecompress()
    
    #self.easyPrintParams()
    #self.iteration +=1

  ########################################################

  def compressOrDecompress(self):
    howFarAlong = ((self.iteration - self.warmupLength) % self.phaseLength) + 1
    print(f"HowFarAlong: {howFarAlong} / {self.phaseLength}")
    print(f"Iteration: {self.iteration}")

    if self.iteration < self.warmupLength:
      ## WARMUP -- PHASE 0
      self.warmup()
      self.notFrozenYet = True

    elif self.iteration >= self.startFineTune:

      if self.notFrozenYet == True:
        ## FREEZING WEIGHTS -- PHASE 1
        self.truncateAndFreeze()
        self.notFrozenYet = False
      else:
        self.compressedStep()
        ## FINE-TUNE
    elif howFarAlong <= self.phaseLength * self.compressionRatio:

      # before it was frozen for an entire epoch
      if self.notFrozenYet == True:
        ## FREEZING WEIGHTS -- PHASE 1
        self.truncateAndFreeze()
        self.notFrozenYet = False
      else:
        ## COMPRESSED -- PHASE 2
        self.compressedStep()

    elif howFarAlong > self.phaseLength * self.compressionRatio:
      ## DECOMPRESS -- PHASE 3
      self.decompressed()
      self.notFrozenYet = True
    else:
      print("Error, iteration logic is incorrect")

  ### PHASES ###
  def warmup(self):
    print('warmup')
    self.updateWeights()
    self.copyGradient()

  def truncateAndFreeze(self):
    print('truncateAndFreeze')
    self.updateWeights()
    self.copyGradient()

    self.sparsify()
    self.freeze()

  def compressedStep(self):
    print('compressed step')
    self.updateWeights()
    self.copyGradient()

    self.refreeze()

  def decompressed(self):
    print('decompressed')
    self.updateWeights()
    self.copyGradient()

  ### UTILITY FUNCTIONS ######################################################################################

  def copyGradient(self):
    #warmup
    if True:
      with torch.no_grad():
        for p in self.paramsIter():
          state = self.state[p]
          state['xt_gradient'] = (p.grad).detach().clone()
        

  def getCutOff(self,sparsity=None,iterate=None):
    if sparsity == None:
      sparsity = self.sparsity
    if iterate == 'zt':
      sparsity = 0.50

    concatWeights = torch.zeros((1)).to(self.device)
    for p in self.paramsIter():
      if iterate == None:
        layer = p.data
      else:
        state = self.state[p]
        layer = state[iterate]

      # CHECK: Make sure this flattening doesn't affect the original layer
      flatWeights = torch.flatten(layer)
      concatWeights = torch.cat((concatWeights,flatWeights),0)
    concatWeights = concatWeights[1:] # Removing first zero

    # Converting the sparsity factor into an integer of respective size
    topK = int(len(concatWeights)*(1-sparsity))

    # All the top-k values are sorted in order, we take the last one as the cutoff
    vals, bestI = torch.topk(torch.abs(concatWeights),topK,dim=0)
    cutoff = vals[-1]

    return cutoff
  
  def sparsify(self,iterate=None):
    cutoff = self.getCutOff(iterate=iterate)

    for p in self.paramsIter():
      state = self.state[p]
      if iterate == None:
        print("!!!!!!!!!!! this should sparsify the params")
        p.data[torch.abs(p) <= cutoff] = 0.0
      else:
        # NOTE: torch.abs(p) is wrong, maybe that's the bug
        (state[iterate])[torch.abs(state[iterate]) <= cutoff] = 0.0
  
  # NOTE: Refreeze is not only for the PARAMS!
  def refreeze(self,iterate=None):
    #print('remember we need to give an iterate for refreeeze')
    for p in self.paramsIter():
      state = self.state[p]
      # TO-DO: make into modular string
      #p.mul_(state['xt_frozen'])
      if iterate == None:
        p.data *= state['xt_frozen']
      else:
        #state[iterate] *= state[f"{iterate}_frozen"]
        state[iterate] *= state['xt_frozen']

  def freeze(self,iterate=None):
    cutOff = self.getCutOff(iterate=iterate)

    for p in self.paramsIter():
      state = self.state[p]
      if iterate == None:
        # NOTE: I CHECKED IT!
        layer = p.data
        state['xt_frozen'] = (torch.abs(layer) > 0).type(torch.uint8)
      else:
        layer = state[iterate]
        state[f"{iterate}_frozen"] = (torch.abs(layer) > 0).type(torch.uint8)

  def trackingSparsity(self):
    concatWeights = torch.zeros((1)).to(self.device)
    concatLinear = torch.zeros((1)).to(self.device)
    concatBias = torch.zeros((1)).to(self.device)
    for layerIdx,layer in enumerate(self.paramsIter()):
      inb = torch.abs(layer.data)

      # Total Weights
      flatTotal = torch.flatten(layer.data)
      concatWeights = torch.cat((concatWeights,flatTotal),0)

      if len(layer.data.shape) < 2:
        # Bias Layers
        concatBias = torch.cat((concatBias,layer.data),0)
      else:
        # Linear Layers
        flatLinear = torch.flatten(layer.data)
        concatLinear = torch.cat((concatLinear,flatLinear),0)

      # Sparsity for this layer
      layerSparsity = torch.mean( (torch.abs(layer.data) > 0).type(torch.float) )
      layerName = f"layerSize{torch.numel(layer)}"
      # NOTE TO SELF: remember, the layer with 10 values isn't strange, it's just the bias layer

      # Track the per-layer sparsity with size
      #self.run[f"trials/{self.trialNumber}/{self.setupID}/{layerName}"].append(layerSparsity)
      #self.run[f"trials/{self.methodName}/sparsities/{layerName}-{layerIdx}{"B" if len(layer.data.shape) < 2 else "L"}"].append(layerSparsity)

    # Removing the First Zero
    print('removed the first zero')
    concatBias = concatBias[1:]
    concatWeights = concatWeights[1:]
    concatLinear = concatLinear[1:] 

    # Final sparsity calculations
    nonZeroWeights = (torch.abs(concatWeights) > 0).type(torch.float)
    nonZeroBias = (torch.abs(concatBias) > 0).type(torch.float)
    nonZeroLinear = (torch.abs(concatLinear) > 0).type(torch.float)

    self.trackSparsity = torch.mean(nonZeroWeights)
    self.trackSparsityBias = torch.mean(nonZeroBias)
    self.trackSparsityLinear = torch.mean(nonZeroLinear)

    ##
    self.run[f"trials/{self.methodName}/sparsities/trackSparsity"].append(self.trackSparsity)
    self.run[f"trials/{self.methodName}/sparsities/trackSparsityBias"].append(self.trackSparsityBias)
    self.run[f"trials/{self.methodName}/sparsities/trackSparsityLinear"].append(self.trackSparsityLinear)
