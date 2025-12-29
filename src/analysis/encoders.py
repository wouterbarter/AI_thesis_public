# analysis/encoders.py
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModel
from typing import List, Dict, Union

class TextEncoder:
    """
    Encodes raw text into numerical features (Sentiment Scores & Dense Embeddings)
    for downstream analysis or regression modeling.
    """
    def __init__(self, 
                 sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english", 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = None):
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.sentiment_model_name = sentiment_model
        self.embedding_model_name = embedding_model
        
        # Lazy loading holders (optional: set to None if you want to load strictly on demand)
        self._sentiment_pipe = None
        self._embed_tokenizer = None
        self._embed_model = None

    @property
    def sentiment_pipe(self):
        if self._sentiment_pipe is None:
            print(f"Loading Sentiment Model ({self.sentiment_model_name})...")
            self._sentiment_pipe = pipeline(
                "sentiment-analysis", 
                model=self.sentiment_model_name, 
                device=0 if self.device == "cuda" else -1
            )
        return self._sentiment_pipe

    @property
    def embed_model(self):
        if self._embed_model is None:
            print(f"Loading Embedding Model ({self.embedding_model_name})...")
            self._embed_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self._embed_model = AutoModel.from_pretrained(self.embedding_model_name).to(self.device)
        return self._embed_model, self._embed_tokenizer

    def get_qa_sentiment(self, qa_pairs: List[Dict[str, str]], strategy: str = "concat") -> List[Dict]:
        """
        Computes sentiment for a list of QA pairs.
        
        Args:
            qa_pairs: List of dicts [{'question': '...', 'answer': '...'}, ...]
            strategy: 'concat' (Q + A) or 'answer_only' (A)
        """
        texts = []
        for p in qa_pairs:
            if strategy == "concat":
                texts.append(f"Q: {p['question']} A: {p['answer']}")
            else:
                texts.append(p['answer'])
        
        # Batch inference via pipeline
        results = self.sentiment_pipe(texts, batch_size=16, truncation=True, max_length=512)

        # return results
        
        # Enrich original dicts with results
        enriched_pairs = []
        for pair, res in zip(qa_pairs, results):
            new_pair = pair.copy()
            new_pair['sentiment_label'] = res['label']
            new_pair['sentiment_score'] = res['score'] 
            enriched_pairs.append(new_pair)
            
        return enriched_pairs

    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Computes dense vector embeddings. Very fast for batch operations.
        Returns: numpy array of shape (n_texts, hidden_dim)
        """
        if isinstance(texts, str): texts = [texts]
        
        model, tokenizer = self.embed_model
        
        # Tokenize
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean Pooling (Attention-Mask aware)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        embeddings = sum_embeddings / sum_mask
        return embeddings.cpu().numpy()