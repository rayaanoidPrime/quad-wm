from pathlib import Path

# from quadwm.models import LatentWorldModel, GenerativeWorldModel TODO
from .checkpt_utils import fetch_checkpoint, load_checkpoint_state_dict, load_pretrained_patch_embed
from .jepawm import JEPAWorldModel


# TODO build model based on the world model type from the config.
def build_model(wm_config: dict, checkpoint_root: str | Path) -> JEPAWorldModel:
    model = JEPAWorldModel(
        embed_dim=wm_config.get("embed_dim", 128),
        image_size=wm_config.get("image_size", 64),
        patch_size=wm_config.get("patch_size", 16),
    )
    init_from = wm_config.get("init_from")
    if init_from:
        # Shape mismatch at toy scale is expected, not fatal -- log and skip
        # rather than crash the smoke run over an optional feature.
        try:
            ckpt_dir = fetch_checkpoint(init_from, checkpoint_root)
            state_dict = load_checkpoint_state_dict(ckpt_dir)
            load_pretrained_patch_embed(model.context_encoder, state_dict, wm_config["init_source_key"])
            print(f"initialized patch embed from {init_from}")
        except Exception as exc:
            print(f"skipping pretrained init from {init_from}: {exc}")
    return model