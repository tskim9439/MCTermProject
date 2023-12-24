from .lstm import LSTM
from .lstm_beta import LSTM_fusion
from .gru_beta import GRU_fusion
from .gru import GRU
from .srgnn import SessionGraph
from .transformer import Transformer
from .transformer_beta import Transformer_fusion


def get_model(cfg, num_classes):
    name = cfg.model.name
    if name == "lstm":
        return LSTM(
            opt=cfg.model.lstm,
            num_classes=num_classes,
        )
    elif name == "lstm_beta":
        return LSTM_fusion(
            opt=cfg.model.lstm_beta,
            num_classes=num_classes,
            n_node=num_classes,
        )
    elif name == "gru":
        return GRU(
            opt=cfg.model.gru,
            num_classes=num_classes,
        )
    elif name == "gru_beta":
        return GRU_fusion(
            opt=cfg.model.gru_beta,
            num_classes=num_classes,
            n_node=num_classes,
        )

    elif name == "transformer":
        return Transformer(
            opt=cfg.model.transformer,
            num_classes=num_classes,
        )
    
    elif name == "transformer_beta":
        return Transformer_fusion(
            opt=cfg.model.gru_beta,
            num_classes=num_classes,
            n_node=num_classes,
        )


    elif name == "srgnn":
        return SessionGraph(
            opt=cfg.model.srgnn,
            n_node=num_classes,
        )

    else:
        raise ValueError(f"Model {name} not supported.")
