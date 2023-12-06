from .lstm import LSTM
from .gru import GRU
from .srgnn import SessionGraph
from .transformer1204 import Transformer


def get_model(cfg, num_classes):
    name = cfg.model.name
    if name == "lstm":
        return LSTM(
            opt=cfg.model.lstm,
            num_classes=num_classes,
        )
    elif name == "gru":
        return GRU(
            opt=cfg.model.gru,
            num_classes=num_classes,
        )
    elif name == "transformer":
        return Transformer(
            opt=cfg.model.transformer,
            num_classes=num_classes,
        )
    elif name == "srgnn":
        return SessionGraph(
            opt=cfg.model.srgnn,
            n_node=num_classes,
        )
    else:
        raise ValueError(f"Model {name} not supported.")
