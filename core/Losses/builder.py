from core.Registry import Registry, build_from_cfg


LOSSES = Registry('losses')

def build_loss(cfg):
    """Build loss."""
    losses = build_from_cfg(cfg, LOSSES)
    return losses