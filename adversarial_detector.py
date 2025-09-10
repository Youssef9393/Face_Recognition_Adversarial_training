import torch

def detect_adversarial(original_tensor, perturbed_tensor, threshold=0.1):
    perturb = perturbed_tensor - original_tensor
    l2_norm = torch.norm(perturb.reshape(perturb.size(0), -1), p=2).item()
    if l2_norm > threshold:
        return True
    else:
        return False
