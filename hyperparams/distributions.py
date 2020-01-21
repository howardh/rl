import torch

class Distribution:
    def __init__(self):
        pass
    def sample(self):
        raise NotImplementedError()

class Uniform(Distribution):
    def __init__(self,min_val,max_val):
        super().__init__()
        assert min_val < max_val
        self.dist = torch.distributions.Uniform(min_val,max_val)
    def sample(self):
        return self.dist.sample().item()

class LogUniform(Uniform):
    def __init__(self,min_val,max_val):
        super().__init__(
                torch.log(torch.tensor(min_val)),
                torch.log(torch.tensor(max_val)))
    def sample(self):
        return torch.exp(self.dist.sample()).item()

class CategoricalUniform(Distribution):
    def __init__(self,vals):
        super().__init__()
        assert len(vals) > 0
        self.vals = vals
        self.dist = torch.distributions.Categorical(
                probs=torch.tensor([1/len(vals)]*len(vals)))
    def sample(self):
        return self.vals[self.dist.sample().item()]

class DiscreteUniform(CategoricalUniform):
    def __init__(self,min_val,max_val):
        assert min_val < max_val
        super().__init__(list(range(min_val,max_val+1)))
