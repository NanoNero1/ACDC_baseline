import torch
from IHT_AGD.optimizers.clipGradientIHTAGD import clipGradientIHTAGD
import numpy as np

class ztSparse_ihtAGD(clipGradientIHTAGD):
  def __init__(self,params,**kwargs):
    super().__init__(params,**kwargs)
    self.methodName = "ztNotSparse_iht_AGD"

  def updateWeightsTwo(self):

    print("AGD updateWeights AND CLIP!!!!")
    # Update z_t the according to the AGD equation in the note
    with torch.no_grad():
        for p in self.paramsIter():

            state = self.state[p]


            #First Get z_t+
            state['zt'] = (state['zt'] - (state['zt_oldGrad'] / self.beta) )

            print('we are heeeeere sparsifying ZT!!!!!')
            self.sparsify('zt')

            # And then we do the actual update, NOTE: zt is actually z_t+ right now
            state['zt'] = (self.sqKappa / (self.sqKappa + 1.0) ) * state['zt'] + (1.0 / (self.sqKappa + 1.0)) * state['xt']

    # CAREFUL! this changes the parameters for the model
    self.getNewGrad('zt')
    self.clipGradients()

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