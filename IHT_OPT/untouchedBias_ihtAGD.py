import torch
#from IHT_AGD.optimizers.ihtAGD import ihtAGD
from IHT_AGD.optimizers.clipGradientIHTAGD import clipGradientIHTAGD
#clipGradientIHTAGD(
import numpy as np

class untouchedBias_ihtAGD(clipGradientIHTAGD):
  def __init__(self,params,**kwargs):
    super().__init__(params,**kwargs)
    self.methodName = "iht_AGD"


  def clipGradients(self,clipAmt=0.01):
    print("I AM CLIPPING!!!!!!")

    #torch.nn.utils.clip_grad_norm_(self.param_groups[0]['params'],norm_type='inf', max_norm=clipAmt)
    torch.nn.utils.clip_grad_value_(self.param_groups[0]['params'],clip_value=clipAmt)
    #for i 
    #torch.clamp(self.param_groups[0]['params'],min=-1.0*clipAmt,clipAmt=1.0)
    pass

  def sparsify(self,iterate=None):
    print('The Bias Nodes should not be sparsified')
    
    #
    cutoff = self.getCutOff(iterate=iterate)

    for p in self.paramsIter():

      if len(p.shape) == 1:
        continue

      state = self.state[p]
      if iterate == None:
        print("!!!!!!!!!!! this should sparsify the params")
        p.data[torch.abs(p) <= cutoff] = 0.0
      else:
        # NOTE: torch.abs(p) is wrong, maybe that's the bug
        (state[iterate])[torch.abs(state[iterate]) <= cutoff] = 0.0
  
  # NOTE: Refreeze is not only for the PARAMS!
  def refreeze(self,iterate=None):
    print('the bias nodes should not be refrozen')
    for p in self.paramsIter():
      if len(p.shape) == 1:
        continue

      state = self.state[p]
      # TO-DO: make into modular string
      #p.mul_(state['xt_frozen'])
      if iterate == None:
        p.data *= state['xt_frozen']
      else:
        state[iterate] *= state[f"{iterate}_frozen"]

