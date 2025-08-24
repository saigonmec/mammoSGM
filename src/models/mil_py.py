import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MILClassifier(nn.Module):
    def __init__(
        self,
        base_model,
        feature_dim,
        num_classes,
        num_patches,
        attention_dim=128,
        nhead=4,
        num_layers=2,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.reduced_dim = feature_dim // 4

        self.tokenizer = nn.Sequential(
            nn.Conv2d(
                feature_dim, self.reduced_dim, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(self.reduced_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.reduced_dim, attention_dim), nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.reduced_dim, attention_dim), nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(attention_dim, 1)

        self.global_processor = nn.Sequential(
            nn.Linear(self.reduced_dim, self.reduced_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(self.reduced_dim + self.reduced_dim // 2, self.reduced_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.reduced_dim, self.reduced_dim // 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.reduced_dim // 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch_size, num_patches+1, C, H, W)
        batch_size, num_patches_plus1, C, H, W = x.size()
        num_patches = num_patches_plus1 - 1
        x_local = x[:, :num_patches]  # (B, N, C, H, W)
        x_global = x[:, num_patches]  # (B, C, H, W)

        # Local patch features
        x_patches = x_local.contiguous().view(-1, C, H, W)
        feat_patches = self.base_model(x_patches)
        if feat_patches.dim() == 2:
            feat_patches = feat_patches.view(
                batch_size * num_patches, self.feature_dim, 1, 1
            )
        tokens_patches = self.tokenizer(feat_patches)
        tokens_patches = tokens_patches.view(batch_size, num_patches, self.reduced_dim)

        # Attention pooling
        A_V = self.attention_V(tokens_patches)
        A_U = self.attention_U(tokens_patches)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 2, 1)
        A = nn.functional.softmax(A, dim=2)
        M = torch.bmm(A, tokens_patches)
        M = M.view(batch_size, -1)

        # Global image features
        global_patch_resized = x_global  # (B, C, H, W)
        feat_global = self.base_model(global_patch_resized)
        if feat_global.dim() == 2:
            feat_global = feat_global.view(batch_size, self.feature_dim, 1, 1)
        tokens_global = self.tokenizer(feat_global)
        tokens_global = tokens_global.view(batch_size, self.reduced_dim)
        processed_global = self.global_processor(tokens_global)

        # Fusion and classification
        fused_features = torch.cat([M, processed_global], dim=1)
        fused_features = self.fusion_layer(fused_features)
        logits = self.classifier(fused_features)
        return logits


class MILClassifierV2(nn.Module):
    def __init__(
        self,
        base_model,
        feature_dim,
        num_classes,
        attn_hidden=256,
        attn_dropout=0.1,
        head_dropout=0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.attn_V = nn.Linear(feature_dim, attn_hidden)
        self.attn_U = nn.Linear(feature_dim, attn_hidden)
        self.attn_w = nn.Linear(attn_hidden, 1)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(head_dropout),
            nn.Linear(feature_dim, num_classes),
        )

        self._init_weights()
        self.last_attn_weights = None

    def _init_weights(self):
        for m in [self.attn_V, self.attn_U, self.attn_w]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def _encode_patches(self, x):
        # x: (B, N, C, H, W)
        B, N, C, H, W = x.shape
        x = x.contiguous().view(B * N, C, H, W)
        feats = self.base_model(x)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        feats = feats.view(B, N, self.feature_dim)
        return feats

    def forward(self, x, mask=None, temperature=1.0):
        # x: (B, N+1, C, H, W)
        B, N_plus1, C, H, W = x.shape
        N = N_plus1 - 1
        x_local = x[:, :N]  # (B, N, C, H, W)
        feats = self._encode_patches(x_local)
        V = torch.tanh(self.attn_V(feats))
        U = torch.sigmoid(self.attn_U(feats))
        scores = self.attn_w(self.attn_drop(V * U)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(scores / max(temperature, 1e-6), dim=1)
        pooled = torch.bmm(attn_weights.unsqueeze(1), feats).squeeze(1)
        logits = self.head(pooled)
        self.last_attn_weights = attn_weights.detach()
        return logits


class _GatedAttnPool(nn.Module):
    """Gated Attention Pooling (compatible with all MILClassifier versions)"""

    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.V = nn.Linear(d_model, hidden)
        self.U = nn.Linear(d_model, hidden)
        self.w = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.constant_(self.V.bias, 0.0)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.constant_(self.U.bias, 0.0)
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.constant_(self.w.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v = torch.tanh(self.V(x))
        u = torch.sigmoid(self.U(x))
        scores = self.w(self.drop(v * u)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores / max(temperature, 1e-6), dim=1)
        pooled = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        return pooled, attn


class MILClassifierV3(nn.Module):
    def __init__(
        self,
        base_model_local: nn.Module,
        base_model_global: nn.Module,
        local_dim: int,
        global_dim: int,
        num_classes: int = 1,
        fusion: str = "concat",
        attn_hidden: int = 256,
        attn_dropout: float = 0.1,
        head_dropout: float = 0.1,
        global_size: tuple = (448, 448),
    ):
        super().__init__()
        assert fusion in ("concat", "gated"), "fusion must be 'concat' or 'gated'"
        self.base_model_local = base_model_local
        self.base_model_global = base_model_global
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.num_classes = num_classes
        self.fusion = fusion
        self.global_size = global_size

        self.mil_pool = _GatedAttnPool(
            d_model=local_dim, hidden=attn_hidden, dropout=attn_dropout
        )

        if fusion == "concat":
            fused_dim = local_dim + global_dim
            self.head = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Dropout(head_dropout),
                nn.Linear(fused_dim, num_classes),
            )
        else:
            self.global_to_local = nn.Linear(global_dim, local_dim)
            self.gate = nn.Sequential(
                nn.Linear(local_dim + local_dim, local_dim),
                nn.ReLU(inplace=True),
                nn.Linear(local_dim, 1),
                nn.Sigmoid(),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(local_dim),
                nn.Dropout(head_dropout),
                nn.Linear(local_dim, num_classes),
            )

        self.last_attn_weights = None
        self.last_global_feat = None
        self.last_local_feat = None

        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        if fusion == "gated":
            nn.init.xavier_uniform_(self.global_to_local.weight)
            nn.init.constant_(self.global_to_local.bias, 0.0)

    def _encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        x_ = x.contiguous().view(B * N, C, H, W)
        feats = self.base_model_local(x_)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        feats = feats.view(B, N, -1)
        return feats

    def _encode_global(self, global_img: torch.Tensor) -> torch.Tensor:
        feats_g = self.base_model_global(global_img)
        if feats_g.dim() == 4:
            feats_g = F.adaptive_avg_pool2d(feats_g, 1).squeeze(-1).squeeze(-1)
        return feats_g

    def forward(self, x_patches: torch.Tensor, mask: torch.Tensor = None):
        # x_patches: (B, N+1, C, H, W)
        # Use first N patches for MIL, last patch as global image
        B, N_plus_1, C, H, W = x_patches.shape
        N = N_plus_1 - 1
        x_local = x_patches[:, :N]  # (B, N, C, H, W)
        x_global = x_patches[:, N]  # (B, C, H, W)

        feats_l = self._encode_patches(x_local)
        pooled_l, attn = self.mil_pool(
            feats_l, mask=mask, temperature=1.0
        )  # (B, feature_dim)
        feats_g = self._encode_global(x_global)  # (B, feature_dim)

        if self.fusion == "concat":
            fused = torch.cat([pooled_l, feats_g], dim=1)  # (B, 2*feature_dim)
            fused = self.head[0](fused)  # LayerNorm
            fused = self.head[1](fused)  # Dropout
            logits = self.head[2](fused)  # Linear
        else:
            g_proj = self.global_to_local(feats_g)
            gate = self.gate(torch.cat([pooled_l, g_proj], dim=1))
            fused = gate * pooled_l + (1.0 - gate) * g_proj
            logits = self.head(fused)

        self.last_attn_weights = attn.detach()
        self.last_global_feat = feats_g.detach()
        self.last_local_feat = pooled_l.detach()
        return logits


class MILClassifierV4(nn.Module):
    """
    MILClassifierV4:
    - Attention pooling cho local patch (MIL)
    - Global feature làm query trong cross-attention, ALL local patches làm key/value
    - Đầu ra fused qua head classifier
    - Sử dụng 1 backbone duy nhất cho cả local và global để chống overfit
    - Thêm biến fusion: 'concat', 'fuse', 'cross_attention'
    """

    def __init__(
        self,
        base_model: nn.Module,
        feature_dim: int,
        num_classes: int = 1,
        attn_hidden: int = 256,
        attn_dropout: float = 0.1,
        head_dropout: float = 0.1,
        cross_attn_heads: int = 4,
        fusion: str = "fuse",  # NEW: fusion method
    ):
        super().__init__()
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.fusion = fusion

        # MIL attention pooling cho local patch
        self.mil_pool = _GatedAttnPool(
            d_model=feature_dim, hidden=attn_hidden, dropout=attn_dropout
        )

        # Only create cross-attention and global_proj if needed
        if fusion == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=feature_dim, num_heads=cross_attn_heads, batch_first=True
            )
            self.global_proj = nn.Linear(feature_dim, feature_dim)
        elif fusion == "concat":
            self.fusion_proj = nn.Sequential(
                nn.LayerNorm(feature_dim * 2),
                nn.Dropout(head_dropout),
                nn.Linear(feature_dim * 2, feature_dim),  # Project to feature_dim
                nn.ReLU(inplace=True),
            )
        elif fusion == "fuse":
            # Adaptive Fusion: learnable gate with softmax over 2 weights
            self.fusion_gate = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, 2),
                nn.Softmax(dim=-1),
            )

        # Head classifier
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(head_dropout),
            nn.Linear(feature_dim, num_classes),
        )

        # Init weights
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        if fusion == "cross_attention":
            nn.init.xavier_uniform_(self.global_proj.weight)
            nn.init.constant_(self.global_proj.bias, 0.0)
        elif fusion == "concat":
            for m in self.fusion_proj:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
        elif fusion == "fuse":
            for m in self.fusion_gate:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)

    def _encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        x_ = x.contiguous().view(B * N, C, H, W)
        feats = self.base_model(x_)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        feats = feats.view(B, N, -1)
        return feats

    def _encode_global(self, global_img: torch.Tensor) -> torch.Tensor:
        feats_g = self.base_model(global_img)
        if feats_g.dim() == 4:
            feats_g = F.adaptive_avg_pool2d(feats_g, 1).squeeze(-1).squeeze(-1)
        return feats_g

    def forward(self, x_patches: torch.Tensor, mask: torch.Tensor = None):
        # x_patches: (B, N+1, C, H, W)
        # Use first N patches for MIL, last patch as global image
        B, N_plus_1, C, H, W = x_patches.shape
        N = N_plus_1 - 1
        x_local = x_patches[:, :N]  # (B, N, C, H, W)
        x_global = x_patches[:, N]  # (B, C, H, W)

        feats_l = self._encode_patches(x_local)  # (B, N, feature_dim)
        pooled_l, attn = self.mil_pool(
            feats_l, mask=mask, temperature=1.0
        )  # (B, feature_dim)
        feats_g = self._encode_global(x_global)  # (B, feature_dim)

        if self.fusion == "cross_attention":
            feats_g_proj = self.global_proj(feats_g).unsqueeze(1)  # (B, 1, feature_dim)
            cross_attn_out, _ = self.cross_attn(
                query=feats_g_proj, key=feats_l, value=feats_l
            )  # (B, 1, feature_dim)
            fused = cross_attn_out.squeeze(1)  # (B, feature_dim)
        elif self.fusion == "concat":
            fused = torch.cat([pooled_l, feats_g], dim=1)  # (B, 2*feature_dim)
            fused = self.fusion_proj(fused)  # (B, feature_dim)
        elif self.fusion == "fuse":
            # Adaptive Fusion: learn weights for local/global, weighted sum
            fusion_input = torch.cat([pooled_l, feats_g], dim=-1)  # (B, 2*feature_dim)
            fusion_weights = self.fusion_gate(fusion_input)  # (B, 2)
            fused = (
                fusion_weights[:, 0:1] * pooled_l + fusion_weights[:, 1:2] * feats_g
            )  # (B, feature_dim)
            self.last_fusion_weights = fusion_weights.detach()
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion}")

        logits = self.head(fused)
        self.last_attn_weights = attn.detach()
        self.last_global_feat = feats_g.detach()
        self.last_local_feat = pooled_l.detach()
        return logits


class MILClassifierV5(nn.Module):
    """
    MILClassifierV5: Advanced MIL classifier with:
    - Dual-path processing (local MIL + global)
    - Cross-attention between global and local features
    - Residual connections and layer normalization
    - Adaptive fusion mechanism
    """

    def __init__(
        self,
        base_model_local: nn.Module,
        base_model_global: nn.Module,
        local_dim: int,
        global_dim: int,
        num_classes: int = 1,
        attn_hidden: int = 256,
        attn_dropout: float = 0.1,
        head_dropout: float = 0.1,
        cross_attn_heads: int = 8,
        fusion_method: str = "adaptive",
    ):
        super().__init__()
        self.base_model_local = base_model_local
        self.base_model_global = base_model_global
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.num_classes = num_classes
        self.fusion_method = fusion_method

        # MIL attention pooling
        self.mil_pool = _GatedAttnPool(
            d_model=local_dim, hidden=attn_hidden, dropout=attn_dropout
        )

        # Project global features to local dimension
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, local_dim),
            nn.LayerNorm(local_dim),
            nn.ReLU(inplace=True),
        )

        # Cross-attention layers
        self.cross_attn_local = nn.MultiheadAttention(
            embed_dim=local_dim, num_heads=cross_attn_heads, batch_first=True
        )
        self.cross_attn_global = nn.MultiheadAttention(
            embed_dim=local_dim, num_heads=cross_attn_heads, batch_first=True
        )

        # Layer norms for residual connections
        self.ln_local = nn.LayerNorm(local_dim)
        self.ln_global = nn.LayerNorm(local_dim)

        # Fusion mechanism
        if fusion_method == "adaptive":
            self.fusion_gate = nn.Sequential(
                nn.Linear(local_dim * 2, local_dim),
                nn.ReLU(inplace=True),
                nn.Linear(local_dim, 2),
                nn.Softmax(dim=-1),
            )
        elif fusion_method == "concat":
            self.fusion_proj = nn.Linear(local_dim * 2, local_dim)

        # Enhanced head with residual connection
        self.head = nn.Sequential(
            nn.LayerNorm(local_dim),
            nn.Linear(local_dim, local_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(local_dim // 2, num_classes),
        )

        self._init_weights()
        self.last_attn_weights = None
        self.last_fusion_weights = None

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        x_ = x.contiguous().view(B * N, C, H, W)
        feats = self.base_model_local(x_)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        feats = feats.view(B, N, -1)
        return feats

    def _encode_global(self, global_img: torch.Tensor) -> torch.Tensor:
        feats_g = self.base_model_global(global_img)
        if feats_g.dim() == 4:
            feats_g = F.adaptive_avg_pool2d(feats_g, 1).squeeze(-1).squeeze(-1)
        return feats_g

    def forward(self, x_patches: torch.Tensor, mask: torch.Tensor = None):
        # x_patches: (B, N+1, C, H, W)
        # Use first N patches for MIL, last patch as global image
        B, N_plus_1, C, H, W = x_patches.shape
        N = N_plus_1 - 1
        x_local = x_patches[:, :N]  # (B, N, C, H, W)
        x_global = x_patches[:, N]  # (B, C, H, W)

        feats_l = self._encode_patches(x_local)
        pooled_l, attn_weights = self.mil_pool(feats_l, mask=mask)  # (B, local_dim)
        feats_g = self._encode_global(x_global)  # (B, global_dim)
        feats_g_proj = self.global_proj(feats_g)  # (B, local_dim)

        # Cross-attention
        # Local-to-global attention
        local_enhanced, _ = self.cross_attn_local(
            query=pooled_l.unsqueeze(1),
            key=feats_g_proj.unsqueeze(1),
            value=feats_g_proj.unsqueeze(1),
        )
        local_enhanced = self.ln_local(pooled_l + local_enhanced.squeeze(1))

        # Global-to-local attention
        global_enhanced, _ = self.cross_attn_global(
            query=feats_g_proj.unsqueeze(1), key=feats_l, value=feats_l
        )
        global_enhanced = self.ln_global(feats_g_proj + global_enhanced.squeeze(1))

        # Fusion
        if self.fusion_method == "adaptive":
            fusion_input = torch.cat([local_enhanced, global_enhanced], dim=-1)
            fusion_weights = self.fusion_gate(fusion_input)  # (B, 2)
            fused = (
                fusion_weights[:, 0:1] * local_enhanced
                + fusion_weights[:, 1:2] * global_enhanced
            )
            self.last_fusion_weights = fusion_weights.detach()
        elif self.fusion_method == "concat":
            fused = self.fusion_proj(
                torch.cat([local_enhanced, global_enhanced], dim=-1)
            )
        else:  # average
            fused = (local_enhanced + global_enhanced) / 2

        # Classification
        logits = self.head(fused)
        self.last_attn_weights = attn_weights.detach()

        return logits


class MILClassifierV6(nn.Module):
    """
    MILClassifierV6: Enhanced MIL classifier based on Local Extremum Mapping (LEM)
    - Incorporates Instance-wise Feature Selection (IFS) and Local Extremum-based Instance Selection (LEIS)
    - Adds Adjacency Matrix-based Spatial Constraint (AMSC) for spatial consistency
    - Maintains dual-path (local MIL + global), cross-attention, and adaptive fusion
    - Outputs classification logits and localization heatmap
    """

    def __init__(
        self,
        base_model_local: nn.Module,
        base_model_global: nn.Module,
        local_dim: int,
        global_dim: int,
        num_classes: int = 1,
        attn_hidden: int = 256,
        attn_dropout: float = 0.1,
        head_dropout: float = 0.1,
        cross_attn_heads: int = 8,
        fusion_method: str = "adaptive",
        top_k: int = 10,  # For LEIS, select top-K instances
        spatial_lambda: float = 0.1,  # Weight for AMSC loss
        grid_size: Tuple[int, int] = (16, 16),  # Grid for spatial adjacency
    ):
        super().__init__()
        self.base_model_local = base_model_local
        self.base_model_global = base_model_global
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.top_k = top_k
        self.spatial_lambda = spatial_lambda
        self.grid_size = grid_size

        # MIL attention pooling (reused for initial scoring)
        self.mil_pool = _GatedAttnPool(
            d_model=local_dim, hidden=attn_hidden, dropout=attn_dropout
        )

        # IFS: Instance-wise Feature Selection
        self.ifs_scorer = nn.Linear(local_dim, 1)  # Score each instance

        # Project global features to local dimension
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, local_dim),
            nn.LayerNorm(local_dim),
            nn.ReLU(inplace=True),
        )

        # Cross-attention layers
        self.cross_attn_local = nn.MultiheadAttention(
            embed_dim=local_dim, num_heads=cross_attn_heads, batch_first=True
        )
        self.cross_attn_global = nn.MultiheadAttention(
            embed_dim=local_dim, num_heads=cross_attn_heads, batch_first=True
        )

        # Layer norms for residual connections
        self.ln_local = nn.LayerNorm(local_dim)
        self.ln_global = nn.LayerNorm(local_dim)

        # Fusion mechanism
        if fusion_method == "adaptive":
            self.fusion_gate = nn.Sequential(
                nn.Linear(local_dim * 2, local_dim),
                nn.ReLU(inplace=True),
                nn.Linear(local_dim, 2),
                nn.Softmax(dim=-1),
            )
        elif fusion_method == "concat":
            self.fusion_proj = nn.Linear(local_dim * 2, local_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(local_dim),
            nn.Linear(local_dim, local_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(local_dim // 2, num_classes),
        )

        # Localization head (for heatmap)
        self.loc_head = nn.Linear(local_dim, 1)  # Score for localization

        self._init_weights()
        self.last_attn_weights = None
        self.last_fusion_weights = None
        self.last_lem_scores = None
        self.last_heatmap = None

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        x_ = x.contiguous().view(B * N, C, H, W)
        feats = self.base_model_local(x_)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        feats = feats.view(B, N, -1)
        return feats

    def _encode_global(self, global_img: torch.Tensor) -> torch.Tensor:
        feats_g = self.base_model_global(global_img)
        if feats_g.dim() == 4:
            feats_g = F.adaptive_avg_pool2d(feats_g, 1).squeeze(-1).squeeze(-1)
        return feats_g

    def _compute_adjacency_matrix(self, B: int, N: int, device) -> torch.Tensor:
        """Compute adjacency matrix for AMSC based on grid positions"""
        H, W = self.grid_size
        if N != H * W:
            # Instead of raising, return None to indicate skip
            return None
        adj = torch.zeros(N, N, device=device)
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                # Connect to 4 neighbors (up, down, left, right)
                if i > 0:
                    adj[idx, (i - 1) * W + j] = 1
                if i < H - 1:
                    adj[idx, (i + 1) * W + j] = 1
                if j > 0:
                    adj[idx, i * W + (j - 1)] = 1
                if j < W - 1:
                    adj[idx, i * W + (j + 1)] = 1
        adj = adj.unsqueeze(0).repeat(B, 1, 1)  # (B, N, N)
        return adj

    def forward(
        self, x_patches: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N_plus_1, C, H, W = x_patches.shape
        N = N_plus_1 - 1
        x_local = x_patches[:, :N]
        x_global = x_patches[:, N]

        # Encode patches and global image
        feats_l = self._encode_patches(x_local)  # (B, N, local_dim)
        feats_g = self._encode_global(x_global)  # (B, global_dim)
        feats_g_proj = self.global_proj(feats_g)  # (B, local_dim)

        # IFS: Instance-wise Feature Selection
        ifs_scores = self.ifs_scorer(feats_l).squeeze(-1)  # (B, N)
        ifs_weights = F.softmax(ifs_scores, dim=1)

        # LEIS: Local Extremum-based Instance Selection
        feats_l_norm = F.normalize(feats_l, dim=-1)
        sim_matrix = torch.einsum(
            "bnd,bmd->bnm", feats_l_norm, feats_l_norm
        )  # (B, N, N)
        k = min(self.top_k, N)
        _, topk_indices = torch.topk(ifs_weights, k=k, dim=1)  # (B, k)
        lem_mask = torch.zeros_like(ifs_weights).scatter_(
            1, topk_indices, 1.0
        )  # (B, N)
        lem_scores = ifs_weights * lem_mask  # Only keep top-K scores
        lem_scores = lem_scores / (lem_scores.sum(dim=1, keepdim=True) + 1e-8)
        pooled_l = torch.sum(
            feats_l * lem_scores.unsqueeze(-1), dim=1
        )  # (B, local_dim)

        # AMSC: Adjacency Matrix-based Spatial Constraint
        adj_matrix = self._compute_adjacency_matrix(B, N, feats_l.device)
        if adj_matrix is not None:
            spatial_loss = self.spatial_lambda * torch.mean(
                torch.sum(
                    adj_matrix
                    * torch.abs(lem_scores.unsqueeze(-1) - lem_scores.unsqueeze(-2)),
                    dim=(1, 2),
                )
            )
        else:
            spatial_loss = torch.tensor(0.0, device=feats_l.device)

        # Cross-attention
        local_enhanced, _ = self.cross_attn_local(
            query=pooled_l.unsqueeze(1),
            key=feats_g_proj.unsqueeze(1),
            value=feats_g_proj.unsqueeze(1),
        )
        local_enhanced = self.ln_local(pooled_l + local_enhanced.squeeze(1))

        global_enhanced, _ = self.cross_attn_global(
            query=feats_g_proj.unsqueeze(1), key=feats_l, value=feats_l
        )
        global_enhanced = self.ln_global(feats_g_proj + global_enhanced.squeeze(1))

        # Fusion
        if self.fusion_method == "adaptive":
            fusion_input = torch.cat([local_enhanced, global_enhanced], dim=-1)
            fusion_weights = self.fusion_gate(fusion_input)  # (B, 2)
            fused = (
                fusion_weights[:, 0:1] * local_enhanced
                + fusion_weights[:, 1:2] * global_enhanced
            )
            self.last_fusion_weights = fusion_weights.detach()
        elif self.fusion_method == "concat":
            fused = self.fusion_proj(
                torch.cat([local_enhanced, global_enhanced], dim=-1)
            )
        else:  # average
            fused = (local_enhanced + global_enhanced) / 2

        # Classification
        logits = self.head(fused)

        # Localization heatmap
        heatmap = self.loc_head(feats_l).squeeze(-1) * lem_scores  # (B, N)
        heatmap = F.normalize(heatmap, dim=1)  # Normalize for visualization

        self.last_lem_scores = lem_scores.detach()
        self.last_heatmap = heatmap.detach()

        return logits  # , spatial_loss, heatmap


class MILClassifierV7(nn.Module):
    """
    MILClassifierSimple: Simplified MIL classifier prioritizing global feature
    - Uses ResNet50 for global feature extraction
    - Selects only the most important local patch per bag
    - Simple concatenation for fusion, no cross-attention or spatial constraints
    - Outputs classification logits only
    """

    def __init__(
        self,
        base_model: nn.Module,  # Single backbone (e.g., ResNet50) for both global and patches
        feature_dim: int,  # Dimension of backbone output (e.g., 2048 for ResNet50)
        num_classes: int = 1,
        dropout: float = 0.3,  # Increased dropout for regularization
        fusion_dim: int = 512,  # Reduced dimension for fusion
    ):
        super().__init__()
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Patch scoring for selecting top-1 patch
        self.patch_scorer = nn.Linear(feature_dim, 1)

        # Projection for global and local features
        self.global_proj = nn.Sequential(
            nn.Linear(feature_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.local_proj = nn.Sequential(
            nn.Linear(feature_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

        self._init_weights()
        self.last_patch_score = None
        self.last_top_patch_idx = None

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        x_ = x.contiguous().view(B * N, C, H, W)
        feats = self.base_model(x_)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        feats = feats.view(B, N, -1)
        return feats

    def _encode_global(self, global_img: torch.Tensor) -> torch.Tensor:
        feats = self.base_model(global_img)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        return feats

    def forward(
        self, x_patches: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N_plus_1, C, H, W = x_patches.shape
        N = N_plus_1 - 1
        x_local = x_patches[:, :N]  # (B, N, C, H, W)
        x_global = x_patches[:, N]  # (B, C, H, W)

        # Encode global feature
        global_feats = self._encode_global(x_global)  # (B, feature_dim)
        global_feats = self.global_proj(global_feats)  # (B, fusion_dim)

        # Encode patches and select top-1
        local_feats = self._encode_patches(x_local)  # (B, N, feature_dim)
        patch_scores = self.patch_scorer(local_feats).squeeze(-1)  # (B, N)
        if mask is not None:
            patch_scores = patch_scores.masked_fill(mask == 0, -1e9)
        top_patch_idx = torch.argmax(patch_scores, dim=1)  # (B,)
        top_patch_feats = local_feats[
            torch.arange(B), top_patch_idx
        ]  # (B, feature_dim)
        top_patch_feats = self.local_proj(top_patch_feats)  # (B, fusion_dim)

        # Fusion: simple concatenation
        fused = torch.cat([global_feats, top_patch_feats], dim=-1)  # (B, fusion_dim)

        # Classification
        logits = self.head(fused)

        self.last_patch_score = patch_scores.detach()
        self.last_top_patch_idx = top_patch_idx.detach()

        return logits


class MILClassifierV8(nn.Module):
    """
    MILClassifierV8: Simple classifier combining global and top-K local patch features
    - Uses ResNet50 for both global and patch feature extraction
    - Selects top-K patches based on linear scoring
    - Fuses global and local features with weighted sum
    - Designed to be lightweight, avoiding MIL pooling or complex attention
    """

    def __init__(
        self,
        base_model: nn.Module,  # ResNet50 backbone
        feature_dim: int,  # Backbone output dim (e.g., 2048 for ResNet50)
        num_classes: int = 1,
        top_k: int = 5,  # Number of top patches to select
        fusion_dim: int = 512,  # Reduced dimension for fusion
        dropout: float = 0.3,  # Dropout for regularization
    ):
        super().__init__()
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.top_k = top_k

        # Patch scoring for top-K selection
        self.patch_scorer = nn.Linear(feature_dim, 1)

        # Projection layers
        self.global_proj = nn.Sequential(
            nn.Linear(feature_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.local_proj = nn.Sequential(
            nn.Linear(feature_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Fusion weights (learnable)
        self.fusion_weights = nn.Parameter(torch.tensor([0.7, 0.3]))  # [global, local]

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes),
        )

        self._init_weights()
        self.last_patch_scores = None
        self.last_topk_indices = None
        self.last_fusion_weights = None

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        nn.init.constant_(self.fusion_weights, 0.5)  # Initialize equal weights

    def _encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        x_ = x.contiguous().view(B * N, C, H, W)
        feats = self.base_model(x_)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        feats = feats.view(B, N, -1)
        return feats

    def _encode_global(self, global_img: torch.Tensor) -> torch.Tensor:
        feats = self.base_model(global_img)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        return feats

    def forward(
        self, x_patches: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N_plus_1, C, H, W = x_patches.shape
        N = N_plus_1 - 1
        x_local = x_patches[:, :N]  # (B, N, C, H, W)
        x_global = x_patches[:, N]  # (B, C, H, W)

        # Encode global feature
        global_feats = self._encode_global(x_global)  # (B, feature_dim)
        global_feats = self.global_proj(global_feats)  # (B, fusion_dim)

        # Encode patches and select top-K
        local_feats = self._encode_patches(x_local)  # (B, N, feature_dim)
        patch_scores = self.patch_scorer(local_feats).squeeze(-1)  # (B, N)
        if mask is not None:
            patch_scores = patch_scores.masked_fill(mask == 0, -1e9)
        k = min(self.top_k, N)
        _, topk_indices = torch.topk(patch_scores, k=k, dim=1)  # (B, k)
        topk_feats = local_feats.gather(
            1, topk_indices.unsqueeze(-1).expand(-1, -1, self.feature_dim)
        )  # (B, k, feature_dim)
        topk_feats = self.local_proj(topk_feats)  # (B, k, fusion_dim)
        local_feats = topk_feats.mean(dim=1)  # (B, fusion_dim)

        # Fusion: weighted sum
        fusion_weights = F.softmax(
            self.fusion_weights, dim=0
        )  # [global_weight, local_weight]
        fused = (
            fusion_weights[0] * global_feats + fusion_weights[1] * local_feats
        )  # (B, fusion_dim)

        # Classification
        logits = self.head(fused)

        self.last_patch_scores = patch_scores.detach()
        self.last_topk_indices = topk_indices.detach()
        self.last_fusion_weights = fusion_weights.detach()

        return logits


class MILClassifierV9(nn.Module):
    """
    MILClassifierV9: Minimal MIL classifier mimicking ResNet50 baseline
    - Prioritizes global feature (90%) like ResNet50 with linear head
    - Uses top-1 patch for minimal local contribution (10%) for paper novelty
    - No dropout, no projection layers, fixed fusion weights
    - Robust for small N (2-5 patches), optimized for fast training
    """

    def __init__(
        self,
        base_model: nn.Module,  # ResNet50 backbone
        feature_dim: int,  # Backbone output dim (e.g., 2048 for ResNet50)
        num_classes: int = 1,
        max_patches: int = 5,  # Max number of patches (N=2-5)
    ):
        super().__init__()
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.max_patches = max_patches

        # Patch scoring for top-1 selection (minimal MIL for paper)
        self.patch_scorer = nn.Linear(feature_dim, 1)

        # Fixed fusion weights (global dominates)
        self.fusion_weights = torch.tensor([0.6, 0.4])  # [global, local]

        # Linear head (like ResNet50 baseline)
        self.head = nn.Linear(feature_dim, num_classes)

        self._init_weights()
        self.last_patch_scores = None
        self.last_top_patch_idx = None
        self.last_fusion_weights = None

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        if N > self.max_patches:
            raise ValueError(
                f"Number of patches ({N}) exceeds max_patches ({self.max_patches})"
            )
        x_ = x.contiguous().view(B * N, C, H, W)
        feats = self.base_model(x_)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        feats = feats.view(B, N, -1)
        return feats

    def _encode_global(self, global_img: torch.Tensor) -> torch.Tensor:
        feats = self.base_model(global_img)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        return feats

    def forward(
        self, x_patches: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N_plus_1, C, H, W = x_patches.shape
        N = N_plus_1 - 1
        x_local = x_patches[:, :N]  # (B, N, C, H, W)
        x_global = x_patches[:, N]  # (B, C, H, W)

        # Encode global feature (main path)
        global_feats = self._encode_global(x_global)  # (B, feature_dim)

        # Encode patches and select top-1 (minimal MIL)
        local_feats = self._encode_patches(x_local)  # (B, N, feature_dim)
        patch_scores = self.patch_scorer(local_feats).squeeze(-1)  # (B, N)
        if mask is not None:
            patch_scores = patch_scores.masked_fill(mask == 0, -1e9)
        top_patch_idx = torch.argmax(patch_scores, dim=1)  # (B,)
        top_patch_feats = local_feats[
            torch.arange(B), top_patch_idx
        ]  # (B, feature_dim)

        # Fusion: fixed weighted sum (global 90%, local 10%)
        fusion_weights = self.fusion_weights.to(global_feats.device)
        fused = (
            fusion_weights[0] * global_feats + fusion_weights[1] * top_patch_feats
        )  # (B, feature_dim)

        # Classification (linear head like ResNet50)
        logits = self.head(fused)

        self.last_patch_scores = patch_scores.detach()
        self.last_top_patch_idx = top_patch_idx.detach()
        self.last_fusion_weights = fusion_weights.detach()

        return logits


class MILClassifierV10(nn.Module):
    """
    MILClassifierV10: Only uses global feature for classification.
    Local features are encoded but not used in prediction.
    """

    def __init__(
        self,
        base_model: nn.Module,  # Backbone (e.g., ResNet50)
        feature_dim: int,  # Output dim of backbone (e.g., 2048)
        num_classes: int = 1,
    ):
        super().__init__()
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.head = nn.Linear(feature_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        # Local features are encoded but not used
        B, N, C, H, W = x.shape
        x_ = x.contiguous().view(B * N, C, H, W)
        feats = self.base_model(x_)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        feats = feats.view(B, N, -1)
        return feats

    def _encode_global(self, global_img: torch.Tensor) -> torch.Tensor:
        feats = self.base_model(global_img)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        return feats

    def forward(
        self, x_patches: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N_plus_1, C, H, W = x_patches.shape
        N = N_plus_1 - 1
        x_local = x_patches[:, :N]  # (B, N, C, H, W)
        x_global = x_patches[:, N]  # (B, C, H, W)

        # Encode local features (not used)
        # _ = self._encode_patches(x_local)

        # Encode global feature (used for prediction)
        global_feats = self._encode_global(x_global)  # (B, feature_dim)
        logits = self.head(global_feats)
        return logits
