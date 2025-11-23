from typing import Optional

from transformers import AutoConfig, AutoModelForTokenClassification
from labels import LABEL2ID, ID2LABEL


def create_model(model_name: str, dropout: Optional[float] = None):
    """
    Create a token-classification model with optional dropout override.
    """
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    if dropout is not None:
        # Align the most common dropout knobs across BERT-style architectures.
        if hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = dropout
        if hasattr(config, "attention_probs_dropout_prob"):
            config.attention_probs_dropout_prob = dropout
        if hasattr(config, "classifier_dropout"):
            config.classifier_dropout = dropout
        if hasattr(config, "dropout"):
            config.dropout = dropout

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
    )
    return model
