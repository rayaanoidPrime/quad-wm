
from pathlib import Path

from huggingface_hub import snapshot_download
import torch


CHECKPOINT_REGISTRY = {
    "vjepa2-vitl": "facebook/vjepa2-vitl-fpc64-256",
    "vjepa2-vith": "facebook/vjepa2-vith-fpc64-256",
    "vjepa2-vitg": "facebook/vjepa2-vitg-fpc64-256",
}


def fetch_checkpoint(name: str, checkpoint_root: str | Path) -> Path:
    """Idempotent: skips the download if the local directory already has files in it."""
    if name not in CHECKPOINT_REGISTRY:
        raise ValueError(f"Unknown checkpoint {name!r}. Known: {list(CHECKPOINT_REGISTRY)}")
    repo_id = CHECKPOINT_REGISTRY[name]
    dest = Path(checkpoint_root) / name
    if dest.exists() and any(dest.iterdir()):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=dest)
    return dest

def load_pretrained_patch_embed(encoder: DepthProprioEncoder, state_dict: dict, source_key: str) -> None:
    """Initialize the depth patch-embed conv from a pretrained RGB (3-channel)
    V-JEPA 2 patch-embed weight, by averaging across the input-channel dim.
 
    `source_key` must be confirmed by inspecting the actual checkpoint's
    state_dict keys first (`list(state_dict.keys())`) -- this has not been
    verified here. V-JEPA 2's patch embed is likely a 3D (video) conv with a
    different kernel shape than this 2D depth patch embed, in which case this
    averaging approach needs adapting (e.g. also collapsing the temporal
    dimension) before it's safe to use. At the toy embed_dim used for the
    smoke test, a shape mismatch against V-JEPA 2's real embed_dim (768+) is
    expected -- this is meant for use once the encoder is scaled to match.
    """
    if source_key not in state_dict:
        raise KeyError(f"{source_key!r} not found in checkpoint -- inspect state_dict.keys() first")
    w = state_dict[source_key]
    if w.shape[1] != 3:
        raise ValueError(f"Expected a 3-channel (RGB) source weight, got shape {tuple(w.shape)}")
    w_gray = w.mean(dim=1, keepdim=True)
    with torch.no_grad():
        if w_gray.shape != encoder.patch_embed.weight.shape:
            raise ValueError(
                f"Shape mismatch after channel-averaging: source {tuple(w_gray.shape)} vs "
                f"target {tuple(encoder.patch_embed.weight.shape)} -- embed_dim/patch_size "
                "likely differ; match them or skip pretrained init at this scale."
            )
        encoder.patch_embed.weight.copy_(w_gray)
 
 
def load_checkpoint_state_dict(ckpt_dir: Path) -> dict:
    """Recent HF checkpoints commonly ship safetensors rather than a .bin --
    try both rather than assuming one, since this hasn't been confirmed for
    the specific V-JEPA 2 repos in checkpoints.py.
    """
    safetensors_path = ckpt_dir / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        return load_file(safetensors_path)
    bin_path = ckpt_dir / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No recognized checkpoint file in {ckpt_dir} -- inspect its contents directly.")