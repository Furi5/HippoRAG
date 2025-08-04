from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

from llama_index.embeddings.ollama import OllamaEmbedding

logger = get_logger(__name__)


class NomicEmbeddingModel(BaseEmbeddingModel):

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name

        self.embedding_model = OllamaEmbedding(
            model_name="nomic-embed-text:latest", 
            base_url="http://localhost:11434"
            )
        self.embedding_dim = 768
        self._init_embedding_config()
        
        
    def _init_embedding_config(self) -> None:
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            
        }
        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)


    def batch_encode(self, texts: List[str], **kwargs) -> None:
        embedding = self.embedding_model.get_text_embedding_batch(texts)
        embedding = np.array(embedding)
        if self.embedding_config.norm:
            embedding = (embedding.T / np.linalg.norm(embedding, axis=1)).T
        return embedding
