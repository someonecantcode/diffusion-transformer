import torch

class EMA:
    """
    https://arxiv.org/pdf/2312.02696
    Section 3.2
    """
    def __init__(self, model, step: int = 1, max_alpha: float = 0.999):
        self.model = model # reference
        self.state_dict = {k: v.clone() for k, v in model.state_dict().items()} # deep copy
        self.step = step
        self.max_alpha = max_alpha
        
    @torch.no_grad()
    def update(self) -> None:
        # alpha * x + (1 - alpha) * new_x
        alpha = min(self.max_alpha, 1 - 1 / self.step)
        for name, param in self.state_dict.items():
            self.state_dict[name] = torch.lerp(self.model.state_dict()[name], param, alpha)
        self.step += 1
    
    def checkpoint(self, file_name: str) -> None:
        torch.save(self.state_dict, file_name)
        
    def load(self, file_name: str) -> None:
        self.state_dict = torch.load(file_name, weights_only=True)