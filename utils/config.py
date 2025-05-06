from yacs.config import CfgNode as CN
import os

_C = CN(new_allowed=True)


def get_cfg_defaults():

    defaults_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'defaults.yaml'))
    _C.merge_from_file(defaults_abspath)
    _C.set_new_allowed(False)
    return _C.clone()


def load_cfg(path=None):
    cfg = get_cfg_defaults()
    if path is not None:
        cfg.merge_from_file(path)
    return cfg


