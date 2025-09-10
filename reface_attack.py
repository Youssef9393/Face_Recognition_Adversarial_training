import torch
import torch.nn.functional as F

def reface_attack(model, source_tensor, target_emb, num_steps=100, lr=0.1, l2_weight=0.001):
    perturbed = source_tensor.clone().detach().to(source_tensor.device).requires_grad_(True)
    optimizer = torch.optim.Adam([perturbed], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        
        emb = model(perturbed)  # Embedding de l’image perturbée
        
        # Similarité cosinus (on veut maximiser)
        cos_sim = F.cosine_similarity(emb, target_emb)
        
        # Loss = on minimise -cos_sim pour maximiser cos_sim
        loss = -cos_sim.mean()
        
        # Régularisation L2 pour limiter perturbation
        perturb = perturbed - source_tensor
        l2_loss = torch.norm(perturb.reshape(perturb.size(0), -1), p=2)
        loss += l2_weight * l2_loss
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            perturbed.clamp_(-1, 1)  # Limiter pixels
        
        if step % 20 == 0 or step == num_steps - 1:
            print(f"Step {step}, Loss: {loss.item():.4f}, Cosine Sim: {cos_sim.item():.4f}, L2: {l2_loss.item():.4f}")
        
        if cos_sim.item() > 0.9:
            print("Succès : attaque réussie, similarité élevée !")
            break

    return perturbed.detach()
