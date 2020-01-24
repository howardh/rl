import numpy as np
import torch

class Distribution:
    def __init__(self):
        pass
    def sample(self):
        raise NotImplementedError()
    def normalize(self,val):
        raise NotImplementedError()
    def unnormalize(self,val):
        raise NotImplementedError()
    def perturb(self,val,scale=0.1):
        raise NotImplementedError()

class Uniform(Distribution):
    def __init__(self,min_val,max_val):
        super().__init__()
        assert min_val < max_val
        self.min_val = min_val
        self.max_val = max_val
        self.dist = torch.distributions.Uniform(min_val,max_val)
    def sample(self):
        return self.dist.sample().item()
    def normalize(self,val):
        # Variance of normal distribution is (b-a)^2/12
        # (b-a)^2/12 = 1
        # b-a = sqrt(12)
        sqrt12 = 3.46410161514
        return (val-self.min_val)/self.max_val*sqrt12-sqrt12/2
    def unnormalize(self,val):
        sqrt12 = 3.46410161514
        return (val+sqrt12/2)/sqrt12*self.max_val+self.min_val
    def perturb(self,val,scale=0.1):
        sqrt12 = 3.46410161514
        val += (np.random.rand()-0.5)*scale*sqrt12
        if val < -sqrt12:
            return -sqrt12
        if val > sqrt12:
            return sqrt12
        return val

class LogUniform(Uniform):
    def __init__(self,min_val,max_val):
        super().__init__(np.log(min_val),np.log(max_val))
    def sample(self):
        return np.exp(self.dist.sample().item())
    def normalize(self,val):
        return super().normalize(np.log(val))
    def unnormalize(self,val):
        return np.exp(super().unnormalize(val))
    def perturb(self,val,scale=0.1):
        return super().perturb(val,scale)

class CategoricalUniform(Distribution):
    def __init__(self,vals):
        super().__init__()
        assert len(vals) > 0
        self.vals = vals
        self.dist = torch.distributions.Categorical(
                probs=torch.tensor([1/len(vals)]*len(vals)))
    def sample(self):
        return self.vals[self.dist.sample().item()]
    def normalize(self,val):
        return val
    def unnormalize(self,val):
        return val
    def perturb(self,val,scale=0.1):
        if np.random.rand() < scale:
            return self.sample()

class DiscreteUniform(Uniform):
    def __init__(self,min_val,max_val):
        assert min_val < max_val
        super().__init__(min_val,max_val+1)
        self.min_val = min_val
        self.max_val = max_val
    def sample(self):
        return int(super().sample())
    def unnormalize(self,val):
        return int(np.rint(super().unnormalize(val)))
    def perturb(self,val,scale=0.1):
        min_scale = 1/(self.max_val-self.min_val)
        scale = max(scale,min_scale)
        return int(np.rint(super().perturb(val,scale)))
