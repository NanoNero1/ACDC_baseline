import torch

class dimitriPytorchSGD(torch.optim.SGD):
    def __init__(self,params,beta=3.0,**kwargs):
        super().__init__(params,lr= (1.0 / beta),**kwargs)
        self.methodName = "Dimitri's Pytorch SGD: so that it works with this framework"