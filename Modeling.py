import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaModel, RobertaPreTrainedModel
import spacy
import textstat
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.stats import pearsonr
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class MultiDimensionalStatisticalProfiling(nn.Module):
    def __init__(self, config, temporal_feat_dim=6):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.temporal_proj = nn.Linear(temporal_feat_dim, 64)
        self.combined_out = nn.Linear(config.hidden_size + 64, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.Tanh()

    def forward(self, text_features, temporal_features):
        x_text = self.dropout(text_features)
        x_text = self.dense(x_text)
        x_text = self.activation(x_text)
        x_temp = self.temporal_proj(temporal_features)
        x_temp = self.activation(x_temp)
        combined = torch.cat([x_text, x_temp], dim=-1)
        return self.combined_out(combined)

class TemporalAnalyzer:
    def __init__(self, window_size=5, stride=1):
        self.window_size = window_size
        self.stride = stride
    
    def _compute_window_stats(self, window):
        window = np.array(window)
        if np.all(window == window[0]):
            return np.array([window[0], 1e-6, 0.0, 0.0])
        
        mean = np.mean(window)
        std = np.std(window) if len(window) > 1 else 1e-6
        autocorr = 0.0
        if len(window) > 1:
            try:
                autocorr = pearsonr(window[:-1], window[1:])[0]
                autocorr = 0.0 if np.isnan(autocorr) else autocorr
            except: 
                pass
        value_range = np.max(window) - np.min(window)
        return np.array([mean, std, autocorr, value_range])
    
    def _create_windows(self, sequence):
        if len(sequence) < self.window_size:
            padding = [sequence[-1]] * (self.window_size - len(sequence))
            return [sequence + padding]
        
        windows = []
        for i in range(0, len(sequence) - self.window_size + 1, self.stride):
            windows.append(sequence[i:i+self.window_size])
        return windows

    def analyze(self, batch_sequences):
        batch_stats = []
        for seq in batch_sequences:
            try:
                syntax = [x[0] for x in seq]
                lexical = [x[1] for x in seq]
                semantic = [x[2] for x in seq]
                
                def process_dim(values):
                    windows = self._create_windows(values)
                    stats = [self._compute_window_stats(w) for w in windows]
                    return np.mean(stats, axis=0) if stats else np.zeros(4)
                
                stats = np.concatenate([
                    process_dim(syntax),
                    process_dim(lexical),
                    process_dim(semantic)
                ])
                batch_stats.append(stats)
            except Exception as e:
                logger.error(f"时序分析错误: {str(e)}")
                raise
        return torch.tensor(batch_stats, dtype=torch.float32)

class RobertaForMultiFacetedContrastiveLearning(RobertaPreTrainedModel):
    def __init__(self, config, temporal_feat_dim=12):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.temporal_feat_dim = temporal_feat_dim
        
        self.roberta = RobertaModel(config)
        self.classifier = MultiDimensionalStatisticalProfiling(config, temporal_feat_dim)
        self.temporal_analyzer = TemporalAnalyzer(window_size=5)
        
        self.contrastive_weight = 0.02
        self.triplet_margin = 0.7
        self.temperature = 0.05
        
        self._nlp = None
        self._semantic_model = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/data/Content_Moderation/model/roberta-base"
        )
        self.init_weights()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    @property
    def semantic_model(self):
        if self._semantic_model is None:
            local_path = "/data/Content_Moderation/all-MiniLM-L6-v2/"
            self._semantic_model = SentenceTransformer(
                local_path,
                device=self.device
            )
        return self._semantic_model

    def _extract_basic_features(self, text):
        doc = self.nlp(text)
        syntax = max([token.head.i for token in doc]) if doc else 0
        lexical = textstat.flesch_kincaid_grade(text)
        
        with torch.no_grad():
            emb = self.semantic_model.encode([text], show_progress_bar=False)
            semantic = 1 - torch.cosine_similarity(
                torch.tensor(emb[:-1]), 
                torch.tensor(emb[1:])
            ).mean().item() if len(emb) > 1 else 0.0
        
        return [syntax, lexical, semantic]

    def _extract_temporal_features(self, input_ids):
        texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        
        batch_sequences = []
        for text in texts:
            doc = self.nlp(text)
            sequence = [self._extract_basic_features(sent.text) for sent in doc.sents]
            batch_sequences.append(sequence)
        
        temporal_stats = self.temporal_analyzer.analyze(batch_sequences)
        return temporal_stats.to(self.device)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        temporal_features=None,
        labels=None,
    ):
        if temporal_features is None:
            temporal_features = self._extract_temporal_features(input_ids)
        else:
            temporal_features = temporal_features.to(self.device)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        text_features = outputs[0][:, 0, :]
        logits = self.classifier(text_features, temporal_features)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                ce_loss = MSELoss()(logits.view(-1), labels.view(-1))
            else:
                ce_loss = CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))

            batch_size = input_ids.size(0)
            group_size = 4
            if batch_size % group_size == 0:
                anchors = text_features[::group_size]
                attacks = text_features[1::group_size]
                positives = text_features[2::group_size]
                negatives = text_features[3::group_size]

                contrastive_loss = self._compute_contrastive_loss(anchors, attacks)
                triplet_loss = self._compute_triplet_loss(anchors, positives, negatives)
                
                total_loss = (1 - self.contrastive_weight) * ce_loss + \
                            self.contrastive_weight * (contrastive_loss + triplet_loss)
                loss = total_loss
            else:
                loss = ce_loss

        return (loss, logits) if loss is not None else logits

    def _compute_contrastive_loss(self, anchors, attacks):
        anchors = F.normalize(anchors, p=2, dim=1)
        attacks = F.normalize(attacks, p=2, dim=1)
        logits = torch.matmul(anchors, attacks.T) / self.temperature
        labels = torch.arange(anchors.size(0), device=self.device)
        return F.cross_entropy(logits, labels)

    def _compute_triplet_loss(self, anchors, positives, negatives):
        pos_dist = F.pairwise_distance(anchors, positives)
        neg_dist = F.pairwise_distance(anchors, negatives)
        return F.relu(pos_dist - neg_dist + self.triplet_margin).mean()

