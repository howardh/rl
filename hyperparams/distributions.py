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

class LogUniform(Uniform):
    def __init__(self,min_val,max_val):
        super().__init__(np.log(min_val),np.log(max_val))
    def sample(self):
        return np.exp(self.dist.sample())
    def normalize(self,val):
        return super().normalize(np.log(val))
    def unnormalize(self,val):
        return np.exp(super().unnormalize(val))

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

class DiscreteUniform(CategoricalUniform):
    def __init__(self,min_val,max_val):
        assert min_val < max_val
        super().__init__(list(range(min_val,max_val+1)))
        self.min_val = min_val
        self.max_val = max_val
    def normalize(self,val):
        sqrt12 = 3.46410161514
        return (val-self.min_val)/self.max_val*sqrt12-sqrt12/2
    def unnormalize(self,val):
        sqrt12 = 3.46410161514
        return int(np.rint((val+sqrt12/2)/sqrt12*self.max_val+self.min_val))
