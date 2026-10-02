"""Simple update attacks for controlled federated robustness experiments."""
import torch


def transform_update(update, global_state, attack, scale=10.0, seed=0):
    """Return a copied update with an attack applied to its model delta."""
    if attack not in {"scale", "sign_flip", "noise"}:
        raise ValueError(f"Unknown attack: {attack}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = {}
    for key, value in update["state_dict"].items():
        base = global_state[key].to(value.device)
        delta = value - base
        if attack == "scale":
            changed = base + scale * delta
        elif attack == "sign_flip":
            changed = base - delta
        else:
            noise = torch.randn(delta.shape, generator=generator,
                                dtype=torch.float32).to(delta.device)
            noise = noise * (delta.float().norm() / max(noise.norm(), 1e-12)) * scale
            changed = base + noise.to(delta.dtype)
        state[key] = changed.detach().clone()
    result = dict(update)
    result["state_dict"] = state
    result["attack"] = attack
    return result
