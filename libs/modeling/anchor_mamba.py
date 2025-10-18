from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'hydra'))
from hydra.modules.hydra import Hydra
from mamba_ssm import Mamba2
from .blocks import TransformerEncoder, MaskedMaxPool1D, SwiGLUFFN
import os

# Set environment variable for triton to use IEEE float32 precision for lower GPU capabilities
# Especially important for using Mamba2 to avoid issues on order GPUs
gpu_id = torch.cuda.current_device()
gpu_capability =torch.cuda.get_device_capability(gpu_id)
if gpu_capability[0] < 8:
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

class BaseAnchorBlock(nn.Module):
    """
    Base class for all anchor blocks containing common anchor generation and interleaving operations.
    
    This base class provides:
    1. Efficient anchor position caching
    2. Robust anchor generation and interleaving
    3. Consistent mask creation and sequence extraction
    4. Configurable pooling methods for anchor generation
    
    Supported pooling methods:
    - "mean": Average pooling (default) - computes mean over sequence blocks
    - "max": Max pooling - takes maximum value over sequence blocks  
    - "attn": Attention pooling - uses learnable query vector similar to CLIP
    - "gated": Gated pooling - adaptive mean/max blending with learnable gates
    
    All derived anchor blocks should inherit from this class to avoid code duplication
    and ensure consistent anchor handling.
    """
    def __init__(self, stride: int, d_model: int, pool_method: str = "mean", dropout: float = 0.1):
        """
        Initialize BaseAnchorBlock with configurable pooling.
        
        Args:
            stride: Stride for downsampling (number of tokens per anchor block)
            d_model: Model embedding dimension
            pool_method: Pooling method for anchor generation. Options:
                        - "mean": Average pooling (default)
                        - "max": Max pooling 
                        - "attn": Attention pooling (CLIP-style with learnable query)
                        - "gated": Gated pooling (learnable mean/max combination)
            dropout: Dropout probability (used for attention pooling)
        """
        super().__init__()
        self.stride = stride
        self.d_model = d_model
        
        # Core anchor pooling component with configurable method
        self.anchor_pooling = AnchorPooling(
            stride=stride, 
            method=pool_method, 
            d_model=d_model, 
            nhead=1, 
            dropout=dropout
        )
        
        # Cache for anchor positions to avoid recomputation
        self._anchor_positions_cache = {}
    
    def _get_anchor_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Cache and retrieve anchor positions to avoid recomputing them each time.
        Args:
            seq_len: Combined sequence length (after interleaving)
        Returns:
            Tensor of anchor positions
        """
        cache_key = (seq_len, str(device))
        if cache_key not in self._anchor_positions_cache:
            self._anchor_positions_cache[cache_key] = torch.arange(
                0, seq_len, self.stride + 1, device=device
            )
        return self._anchor_positions_cache[cache_key]

    def _generate_and_interleave_anchors(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized method to interleave anchors with sequence tokens.

        Args:
            x: Input tensor of shape (B, D, L)
            mask: Input mask of shape (B, 1, L)

        Returns:
            Tuple of:
            - x_combined: Interleaved anchors and sequences (B, combined_seq_len, D)
            - anchor_positions: Positions of anchors in combined sequence
            - expanded_mask: Mask for combined sequence (B, combined_seq_len, 1)
            - anchor_mask: Downsampled mask for anchors (B, 1, num_blocks)
        """
        B, D, L = x.shape
        s = self.stride
        device = x.device

        # 1) Correctly calculate number of blocks and required padding
        num_blocks = (L + s - 1) // s
        needed_len = num_blocks * s
        pad_len = needed_len - L

        if pad_len > 0:
            # Pad with zeros to make the sequence length a multiple of the stride
            x_padded = F.pad(x, (0, pad_len))
        else:
            x_padded = x

        # 2) Reshape into blocks of size <s> for pooling AND interleaving
        x_blocks = x_padded.reshape(B, D, num_blocks, s)

        # 3) Generate anchors using the configurable pooling method
        anchors = self.anchor_pooling(x_padded)
        
        # 4) Vectorized interleaving: Concatenate anchors and token blocks
        #    First, add a dimension to anchors to enable concatenation.
        anchors_expanded = anchors.unsqueeze(-1)  # (B, D, num_blocks, 1)
        mixed = torch.cat((anchors_expanded, x_blocks), dim=-1) # (B, D, num_blocks, s+1)

        # 5) Flatten back into a single sequence and permute to (B, L', D)
        x_combined = mixed.permute(0, 2, 3, 1).reshape(B, -1, D)

        # 6) Get anchor positions using the cached helper method
        anchor_positions = self._get_anchor_positions(x_combined.size(1), device)

        # 7) Create the correct masks for the new packed sequence
        anchor_mask = downsample_mask(mask, s)  # (B, 1, num_blocks)

        nonmasked_seq_len = int(mask.sum().item())
        nonmasked_anchor_len = int(anchor_mask.sum().item())
        nonmasked_len = nonmasked_seq_len + nonmasked_anchor_len

        # Create a packed mask for the combined sequence. This assumes all
        # valid tokens/anchors will be packed at the start of the sequence
        # in a subsequent operation (common for attention).
        expanded_mask = torch.zeros((B, x_combined.size(1), 1), dtype=torch.bool, device=device)
        expanded_mask[:, :nonmasked_len] = True

        return x_combined, anchor_positions, expanded_mask, anchor_mask

    def _generate_and_interleave_anchors_initial_v2(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Interleave an averaged anchor in front of every <stride> tokens:
            [a0, t0, t1,  a1, t2, t3,  ...]
        """
        B, D, L = x.shape
        s       = self.stride                    # e.g. 2
        device  = x.device

        # 1) Pad once so that L % s == 0  (keeps pooling semantics exact)
        pad_len = (-L) % s
        if pad_len:
            x = F.pad(x, (0, pad_len))           # (B, D, L+pad)

        # 2) Tokens grouped in blocks of length <s>
        num_blocks = x.size(-1) // s
        x_blocks   = x.view(B, D, num_blocks, s) # contiguous thanks to the pad

        # 3) Anchors via cuDNN avg_pool1d  (kernel = stride = s)
        #    shape comes out (B, D, num_blocks) → add a length-1 dim to match x_blocks
        anchors = F.avg_pool1d(x, kernel_size=s, stride=s)     # (B, D, num_blocks)
        anchors = anchors.unsqueeze(-1)                        # (B, D, num_blocks, 1)

        # 4) Concatenate anchor + tokens  →  (B, D, num_blocks, s+1)
        mixed = torch.cat((anchors, x_blocks), dim=-1)

        # 5) Flatten back to sequence, return (B, seq, D)
        x_combined = mixed.permute(0, 2, 3, 1).reshape(B, -1, D)

        # 6) Anchor positions are every (s+1) step
        anchor_positions = torch.arange(0, x_combined.size(1), s+1, device=device)

        # 7) Masks – stay compact
        anchor_mask = downsample_mask(mask, s)             # (B,1,num_blocks)

        seq_mask = mask.squeeze(1)                         # (B,L)
        if pad_len:
            seq_mask = F.pad(seq_mask, (0, pad_len), value=False)
        seq_mask = seq_mask.view(B, num_blocks, s)         # (B,blk,s)
           
        combined_seq_len = x_combined.size(1)
        # Create extended mask for combined sequence
        nonmasked_seq_len = int(mask.sum().item())
        nonmasked_anchor_len = int(anchor_mask.sum().item())
        nonmasked_len = nonmasked_seq_len + nonmasked_anchor_len
        # (B, T′)
        expanded_mask = torch.zeros((B, combined_seq_len, 1), dtype=torch.bool, device=device)
        expanded_mask[:, :nonmasked_len] = True
        return x_combined, anchor_positions, expanded_mask, anchor_mask
        
    def _generate_and_interleave_anchors__initial(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate anchors and interleave them with sequence tokens.

        Args:
            x: Input tensor of shape (B, D, L)
            mask: Input mask of shape (B, 1, L)
            
        Returns:
            Tuple of:
            - x_combined: Interleaved anchors and sequences (B, combined_seq_len, D)
            - anchor_positions: Positions of anchors in combined sequence
            - expanded_mask: Mask for combined sequence (B, combined_seq_len, 1)
            - anchor_mask: Downsampled mask for anchors (B, 1, num_blocks)
        """
        B, D, L = x.shape
        device = x.device
        
        # Downsample input mask for anchors
        anchor_mask = downsample_mask(mask, self.stride)
        
        # 1) Compute how many anchor-blocks we have
        num_blocks = (L + self.stride - 1) // self.stride
        
        # 2) Pad & block into shape (B, D, num_blocks, stride)
        needed_len = num_blocks * self.stride
        pad_len = needed_len - L
        
        # Avoid redundant copying when no padding is needed
        if pad_len > 0:
            x_padded = F.pad(x, (0, pad_len))
        else:
            x_padded = x
            
        # Use contiguous before reshape for better memory layout
        x_reshaped = x_padded.reshape(B, D, num_blocks, self.stride)
        
        # 3) Extract initial anchors
        anchors = self.anchor_pooling(x_reshaped)  # → (B, D, num_blocks)
        
        # Calculate combined sequence length once
        combined_seq_len = num_blocks * (self.stride + 1)
        
        # More efficient interleaving with fewer intermediate tensors
        # Pre-allocate the output tensor directly
        x_combined = torch.empty((B, combined_seq_len, D), dtype=x.dtype, device=device)
        
        # Place anchors at stride+1 intervals
        anchor_positions = self._get_anchor_positions(combined_seq_len, device)
        x_combined[:, anchor_positions] = anchors.permute(0, 2, 1)  # (B, num_blocks, D)
        
        # Place sequence tokens efficiently
        for i in range(self.stride):
            seq_positions = anchor_positions + i + 1
            # Handle boundary conditions
            valid_mask = seq_positions < combined_seq_len
            if not valid_mask.all():
                seq_positions = seq_positions[valid_mask]
                seq_blocks = x_reshaped[:, :, :seq_positions.size(0), i]
            else:
                seq_blocks = x_reshaped[:, :, :, i]
            x_combined[:, seq_positions] = seq_blocks.permute(0, 2, 1)  # (B, blocks, D)
        
        # Create extended mask for combined sequence
        nonmasked_seq_len = int(mask.sum().item())
        nonmasked_anchor_len = int(anchor_mask.sum().item())
        nonmasked_len = nonmasked_seq_len + nonmasked_anchor_len
        
        # Apply mask directly to avoid extra operations
        expanded_mask = torch.zeros((B, combined_seq_len, 1), dtype=torch.bool, device=device)
        expanded_mask[:, :nonmasked_len] = True
        
        return x_combined, anchor_positions, expanded_mask, anchor_mask

    def _extract_anchor_and_sequence_outputs(
            self,
            processed_features: torch.Tensor,      # (B, combined_seq_len, D)
            anchor_positions: torch.Tensor,
            original_seq_len: int                  # L (without padding)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Faster, allocation-free extraction of anchors and sequence tokens.
        """
        B, combined_len, D = processed_features.shape
        s        = self.stride                    # stride used during interleave
        step     = s + 1                          # anchor + <s> tokens
        device   = processed_features.device

        # 1) Reshape once so we can index by block and within-block position.
        num_blocks = combined_len // step
        mixed      = processed_features.view(B, num_blocks, step, D)  # no copy

        # 2) Anchors live at position 0 in every block.
        #    Shape wanted: (B, D, num_blocks)
        anchor_out = mixed[:, :, 0, :].transpose(1, 2)                # view

        # 3) Sequence tokens are positions 1..s in every block.
        #    Flatten them back to a 1-D temporal axis, trim to `L`.
        seq_tokens = mixed[:, :, 1:, :].reshape(B, num_blocks * s, D) # view
        seq_out    = seq_tokens.transpose(1, 2)[..., :original_seq_len]

        return anchor_out, seq_out
    
    def _extract_anchor_and_sequence_outputs_initial(self, processed_features: torch.Tensor, anchor_positions: torch.Tensor, original_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract anchor and sequence outputs from processed combined features.
        
        Args:
            processed_features: Processed combined features (B, combined_seq_len, D)
            anchor_positions: Positions of anchors in combined sequence
            original_seq_len: Original sequence length L
            
        Returns:
            Tuple of:
            - anchor_out: Extracted anchor features (B, D, num_blocks)
            - seq_out: Extracted sequence features (B, D, L)
        """
        device = processed_features.device
        
        # Extract anchor tokens
        updated_anchors = processed_features.index_select(1, anchor_positions)  # (B, num_blocks, D)
        anchor_out = updated_anchors.permute(0, 2, 1)  # (B, D, num_blocks)
        
        # Extract sequence tokens efficiently
        seq_mask = torch.ones(processed_features.size(1), dtype=torch.bool, device=device)
        seq_mask.index_fill_(0, anchor_positions, False)
        seq_indices = seq_mask.nonzero(as_tuple=True)[0]
        
        updated_seq = processed_features.index_select(1, seq_indices)  # (B, seq_len, D)
        seq_out = updated_seq.permute(0, 2, 1)[:, :, :original_seq_len]  # (B, D, L)
        
        return anchor_out, seq_out

class AnchorMambaPoolingBlockGated(BaseAnchorBlock):
    """
    Anchor Mamba Pooling Block with Gated Fusion
    
    This block combines the refined gating mechanism 
    with the optional local encoder pattern from AnchorMambaPoolingBlock.
    
    Key Features:
    1. Hierarchical gated fusion (gate1 always, gate2 only when local encoder enabled)
    2. Optional local encoder (can be disabled)
    3. Final FFN always present (not optional)
    4. Strong residual connections throughout
    
    Architecture:
    When local_encode=True:
    input → global → gate1 → local → gate2 → ffn → output
                     ↓               ↓       ↓
                     └─── residual ──┴───────┘
    
    When local_encode=False:
    input → global → gate1 → ffn → output
                     ↓       ↓
                     └───────┘
    
    Flow:
    1. Global encoding with residual connection
    2. First gated fusion (input + global)
    3. Optional local encoding + second gated fusion (if local_encode=True)
    4. Final FFN (always applied)
    """
    def __init__(
        self, 
        stride: int = 2, 
        d_model: int = 384, 
        nhead: int = 4, 
        local_window_size: int = 5,
        dropout: float = 0.1, 
        ffn_ratio: int = 4,
        local_encode: bool = False,  # Optional local encoder
        pool_method: str = "mean",
        local_encoder_type: str = "transformer",
        mamba_headdim: int = 64, 
        mamba_dstate: int = 64,
        mamba_expand: int = 2,
        mamba_dconv: int = 7,
        bidirectional: bool = True
    ):
        super().__init__(stride=stride, d_model=d_model, pool_method=pool_method, dropout=dropout)
        
        self.local_encode = local_encode
        
        if bidirectional:
            # Global encoder block 
            self.global_encoder = Hydra(
                d_model=d_model, d_state=mamba_dstate, d_conv=mamba_dconv, expand=mamba_expand,
                use_mem_eff_path=True, headdim=mamba_headdim
            )
        else:
            self.global_encoder = Mamba2(
                d_model=d_model, d_state=mamba_dstate, d_conv=4, expand=mamba_expand,
                headdim=48
            )
        self.norm_global = RMSNorm(d_model)
        self.drop_path_global = LayerScale2(d_model, dropout)

        # Hierarchical gating mechanisms
        self.gate1 = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        
        # Optional local encoder components
        if local_encode:
            # Gate2 only needed when local encoder is enabled
            self.gate2 = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.Sigmoid()
            )
            self.norm_local = RMSNorm(d_model)
            self.drop_path_local = LayerScale2(d_model, dropout)
            
            # Choose local encoder based on type parameter
            if local_encoder_type == "mamba":
                self.local_encoder = MambaWindowEncoder(
                    embd_dim=d_model,
                    stride=1, 
                    window_size=local_window_size,
                    n_heads=nhead // 2,
                    head_dim=mamba_headdim,
                    dropout=dropout,
                    path_pdrop=dropout
                )
            elif local_encoder_type == "dwconv":
                self.local_encoder = LocalDWConvGLU(d_model, expansion=1)
            else:  # default to transformer
                self.local_encoder = TransformerEncoder(
                    d_model, 
                    stride=1, 
                    window_size=local_window_size, 
                    n_heads=2
                )
            self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
        else:
            self.ffn = SwiGLUFFN(d_model, dropout=dropout)

        self.norm_ffn = RMSNorm(d_model)
        self.drop_path_ffn = LayerScale2(d_model, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, L = x.shape
        device = x.device

        if mask is None:
            mask = torch.ones((B, 1, L), dtype=torch.bool, device=device)

        x.masked_fill_(~mask, 0.)  # In-place masking saves memory

        # Generate and interleave anchors
        x_combined, anchor_positions, expanded_mask, anchor_mask = self._generate_and_interleave_anchors(x, mask)

        # Global encoding with residual
        global_out = self.global_encoder(self.norm_global(x_combined))
        global_out = self.drop_path_global(global_out) + x_combined
        global_out.mul_(expanded_mask)  # In-place masking

        # First gated fusion
        gate1_weights = self.gate1(torch.cat([x_combined, global_out], dim=-1))
        fusion1_out = gate1_weights * global_out + (1 - gate1_weights) * x_combined

        # Optional local encoding and second fusion
        if self.local_encode:
            local_out = self.local_encoder(
                self.norm_local(fusion1_out).transpose(1, 2),
                expanded_mask.transpose(1, 2)
            )[0].transpose(1, 2)
            local_out = self.drop_path_local(local_out) + fusion1_out
            local_out.mul_(expanded_mask)

            gate2_weights = self.gate2(torch.cat([fusion1_out, local_out], dim=-1))
            fused = gate2_weights * local_out + (1 - gate2_weights) * fusion1_out
        else:
            fused = fusion1_out  # skip local encoding

        # Final FFN with residual
        ffn_out = self.ffn(self.norm_ffn(fused))
        final_out = self.drop_path_ffn(ffn_out) + fused

        # Extract outputs
        anchor_out, seq_out = self._extract_anchor_and_sequence_outputs(final_out, anchor_positions, L)
        return anchor_out, seq_out, anchor_mask, mask
        
    def forward_before(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, L = x.shape
        device = x.device

        if mask is None:
            mask = torch.ones((B, 1, L), dtype=torch.bool, device=device)
        
        x.masked_fill_(~mask, 0.)  # In-place masking

        # Generate and interleave anchors
        x_combined, anchor_positions, expanded_mask, anchor_mask = self._generate_and_interleave_anchors(x, mask)
        
        # Store original input for gating
        original_features = x_combined
        
        # Step 1: Global encoding with residual connection
        global_norm = self.norm_global(x_combined)
        global_out = self.global_encoder(global_norm)
        global_out = x_combined + self.drop_path_global(global_out)
        global_out = global_out * expanded_mask
        
        # Step 2: First gated fusion (input + global)
        gate1_input = torch.cat([original_features, global_out], dim=-1)
        gate1_weights = self.gate1(gate1_input)
        fusion1_out = gate1_weights * global_out + (1 - gate1_weights) * original_features
        
        # Step 3: Optional local encoding and gating
        if self.local_encode:
            local_norm = self.norm_local(fusion1_out)
            local_transposed = local_norm.transpose(1, 2).contiguous()
            # encoder_mask = expanded_mask.transpose(1, 2)
            local_out_transposed = self.local_encoder(local_transposed, expanded_mask.transpose(1, 2))[0]
            local_out = local_out_transposed.transpose(1, 2)
            local_out = fusion1_out + self.drop_path_local(local_out)
            local_out = local_out * expanded_mask
            
            # Step 4: Second gated fusion (fusion1 + local)
            gate2_input = torch.cat([fusion1_out, local_out], dim=-1)
            gate2_weights = self.gate2(gate2_input)
            gate2_out = gate2_weights * local_out + (1 - gate2_weights) * fusion1_out
        else:
            # No local encoder - skip gate2 entirely 
            gate2_out = fusion1_out
        
        # Step 5: Final FFN (always applied)
        ffn_norm = self.norm_ffn(gate2_out)
        ffn_out = self.ffn(ffn_norm)
        final_out = gate2_out + self.drop_path_ffn(ffn_out)
        
        # Extract anchor and sequence outputs
        anchor_out, seq_out = self._extract_anchor_and_sequence_outputs(
            final_out, anchor_positions, L
        )
        
        return anchor_out, seq_out, anchor_mask, mask

class AnchorMambaPoolingBlock(BaseAnchorBlock):
    """
    A two-stage global-local encoder block for sequence modeling that inherits from BaseAnchorBlock.
    
    This class use the BaseAnchorBlock's robust anchor generation and output extraction methods.

    Input:
        x:    Tensor of shape (B, D, L), where
              B = batch size,
              D = embedding dimension,
              L = sequence length (can be odd).
        mask: Tensor of shape (B, 1, L), optional boolean mask.

    Output:
        anchor_out: (B, D, ceil(L / stride)) - downsampled anchors
        seq_out:    (B, D, L)  -- same length as input (padding removed)
        anchor_mask: downsampled version of input mask
        mask: original input mask
    """
    def __init__(
        self, 
        stride: int, 
        d_model: int, 
        nhead: int = 4, 
        local_window_size: int = 5,
        dropout: float = 0.1, 
        ffn_ratio: int = 4,
        use_swiglu: bool = True,
        local_encode: bool = False, 
        pool_method: str = "mean",
        local_encoder_type: str = "transformer",  # Options: 'transformer' or 'mamba',
        mamba_headdim: int = 64, 
        mamba_dstate: int = 64,
        mamba_expand: int = 2,
        mamba_dconv: int = 7,
        bidirectional: bool = True
    ):
        super().__init__(stride=stride, d_model=d_model, pool_method=pool_method, dropout=dropout)
        
        self.local_encode = local_encode

        if bidirectional:
            # Global encoder block 
            self.global_encoder = Hydra(
                d_model=d_model, d_state=mamba_dstate, d_conv=mamba_dconv, expand=mamba_expand,
                use_mem_eff_path=True, headdim=mamba_headdim
            )
        else:
            self.global_encoder = Mamba2(
                d_model=d_model, d_state=mamba_dstate, d_conv=4, expand=mamba_expand,
                headdim=48
            )
        self.norm_global_in = RMSNorm(d_model)
        self.drop_path_global_1 = LayerScale2(d_model, dropout)

        self.norm_ffn_in = RMSNorm(d_model)
        self.drop_path_ffn = LayerScale2(d_model, dropout)

        if local_encode:
            self.norm_local_in = RMSNorm(d_model)
            self.drop_path_local = LayerScale2(d_model, dropout)
            # Choose local encoder based on type parameter
            if local_encoder_type == "dwconv":
                self.local_encoder = LocalDWConvGLU(d_model, expansion=1)
            else:  # default to transformer
                self.local_encoder = TransformerEncoder(
                    d_model, 
                    stride=1, 
                    window_size=local_window_size, 
                    n_heads=2
                )
            self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
        else:
            self.ffn = SwiGLUFFN(d_model, dropout=dropout) if use_swiglu else nn.Sequential(
                nn.Linear(d_model, ffn_ratio * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_ratio * d_model, d_model),
                nn.Dropout(dropout)
            )

    def forward(                     # ── (B , D , L)
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        B, D, L  = x.shape
        device   = x.device
        s        = self.stride                     # cache once

        # ---------------------------------------------------------------------
        # 0) Ensure mask exists and zero-out padded / invalid tokens *in-place*
        # ---------------------------------------------------------------------
        if mask is None:
            mask = torch.ones((B, 1, L), dtype=torch.bool, device=device)

        x.masked_fill_(~mask, 0.)                  # no new allocation

        # ---------------------------------------------------------------------
        # 1) Interleave anchors  (compact expanded_mask is shape (B, T′) bool)
        # ---------------------------------------------------------------------
        x_combined, anchor_pos, expanded_mask, anchor_mask = \
            self._generate_and_interleave_anchors(x, mask)          # T′ = (s+1)·blocks
        # x_combined : (B, T′, D)     expanded_mask : (B, T′)  bool

        # ---------------------------------------------------------------------
        # 2) Global encoder (Hydra)  —  prenorm + residual
        # ---------------------------------------------------------------------
        g = self.norm_global_in(x_combined)
        g = self.global_encoder(g)                 # (B, T′, D)
        g = g + self.drop_path_global_1(x_combined)   # in-place add if LayerScale2 uses .add_

        # Zero out invalid slots again (they may have acquired values)
        g.masked_fill_(~expanded_mask, 0.)

        # ---------------------------------------------------------------------
        # 3) Optional local encoder  (single transpose pair, mask reused)
        # ---------------------------------------------------------------------
        if self.local_encode:
            l = self.norm_local_in(g)

            l_t = l.transpose(1, 2).contiguous()        # (B, D, T′)
            # local encoder expects key-padding mask shape  (B, 1, T′)
            encoder_mask = expanded_mask.transpose(1, 2)  # (B, 1, T′)
            l_t = self.local_encoder(l_t, encoder_mask)[0]    # (B, D, T′)

            l   = l_t.transpose(1, 2)                    # (B, T′, D)
            g   = g + self.drop_path_local(l)            # residual
        # else: g already holds the right value

        # ---------------------------------------------------------------------
        # 4) Feed-forward network (always applied)
        # ---------------------------------------------------------------------
        z  = self.norm_ffn_in(g)
        z  = self.ffn(z)                                 # (B, T′, D)
        z  = g + self.drop_path_ffn(z)                   # residual

        # ---------------------------------------------------------------------
        # 5) Split back into anchor / sequence streams (view-only extraction)
        # ---------------------------------------------------------------------
        anchor_out, seq_out = self._extract_anchor_and_sequence_outputs(
            z,                                            # (B, T′, D)
            anchor_pos,                                   # anchor positions
            L                                             # trims padding
        )

        return anchor_out, seq_out, anchor_mask, mask



class EnhancedAnchorBlock_Refined(BaseAnchorBlock):
    """
    Enhanced Anchor Block - Refined Gating with Optional Enhancement
    
    Based on the insight that residual connections to gated features are critical,
    this block explores different ways to enhance gated features:
    1. Optional FFN refinement
    2. Lightweight enhancement layers
    3. Strong residual connections to preserve gated information
    
    Architecture:
    input → global → gate1 → local → gate2 → [optional_refinement] → output
                                     ↓                ↑
                                     └─── residual ───┘
    """
    def __init__(
        self, 
        stride: int, 
        d_model: int, 
        nhead: int = 4, 
        local_window_size: int = 19,
        dropout: float = 0.1, 
        ffn_ratio: int = 4,
        pool_method: str = "mean",
        local_encode: bool = True,  # Optional local encoder
        local_encoder_type: str = "transformer",
        use_ffn_refinement: bool = True,
        refinement_type: str = "lightweight",  # Options: 'lightweight', 'full_ffn', 'none',
        mamba_headdim: int = 64, 
        mamba_dstate: int = 64,
        mamba_expand: int = 2,
        mamba_dconv: int = 7,
        bidirectional: bool = True
    ):
        super().__init__(stride=stride, d_model=d_model, pool_method=pool_method, dropout=dropout)

        # Core processing components
        self.global_encoder = Hydra(d_model=d_model, d_state=mamba_dstate, d_conv=mamba_dconv, expand=mamba_expand, use_mem_eff_path=True, headdim=mamba_headdim)
        
        if local_encoder_type == "mamba":
            self.local_encoder = MambaWindowEncoder(embd_dim=d_model, stride=1, window_size=local_window_size, n_heads=nhead//2, head_dim=mamba_headdim, dropout=dropout, path_pdrop=dropout)
        else:
            self.local_encoder = TransformerEncoder(d_model, stride=1, window_size=local_window_size, n_heads=2)
        
        # Hierarchical gating mechanisms (same as successful versions)
        self.gate1 = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        
        self.gate2 = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        
        # Refinement mechanism based on type
        self.use_ffn_refinement = use_ffn_refinement
        self.refinement_type = refinement_type
        
        if use_ffn_refinement:
            if refinement_type == "lightweight":
                # Lightweight refinement - just a single linear layer with activation
                self.refinement = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            elif refinement_type == "full_ffn":
                # Full FFN refinement
                self.refinement = nn.Sequential(
                    nn.Linear(d_model, ffn_ratio * d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_ratio * d_model, d_model),
                    nn.Dropout(dropout)
                )
            elif refinement_type == "conv_refinement":
                # Convolutional refinement for local patterns
                self.refinement = nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),  # Depthwise conv
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(d_model, d_model, kernel_size=1),  # Pointwise conv
                    nn.Dropout(dropout)
                )
        
        # Normalization layers
        self.norm_global = RMSNorm(d_model)
        self.norm_local = RMSNorm(d_model)
        if use_ffn_refinement:
            self.norm_refinement = RMSNorm(d_model)
        
        # Drop path layers
        self.drop_path_1 = LayerScale2(d_model, dropout)
        self.drop_path_2 = LayerScale2(d_model, dropout)
        if use_ffn_refinement:
            self.drop_path_3 = LayerScale2(d_model, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, L = x.shape
        device = x.device
        
        if mask is None:
            mask = torch.ones((B, 1, L), dtype=torch.bool, device=device)
        x = x * mask.to(x.dtype)

        # Use base class method for anchor generation and interleaving
        x_combined, anchor_positions, expanded_mask, anchor_mask = self._generate_and_interleave_anchors(x, mask)
        
        # Store original input
        original_features = x_combined
        
        # Step 1: Global encoding
        global_norm = self.norm_global(x_combined)
        global_out = self.global_encoder(global_norm)
        global_out = x_combined + self.drop_path_1(global_out)
        global_out = global_out * expanded_mask
        
        # Step 2: First gated fusion (input + global)
        gate1_input = torch.cat([original_features, global_out], dim=-1)
        gate1_weights = self.gate1(gate1_input)
        fusion1_out = gate1_weights * global_out + (1 - gate1_weights) * original_features
        
        # Step 3: Local encoding (processes fusion1 result)
        local_norm = self.norm_local(fusion1_out)
        local_transposed = local_norm.transpose(1, 2)
        encoder_mask = expanded_mask.transpose(1, 2)
        local_out_transposed = self.local_encoder(local_transposed, encoder_mask)[0]
        local_out = local_out_transposed.transpose(1, 2)
        local_out = fusion1_out + self.drop_path_2(local_out)
        local_out = local_out * expanded_mask
        
        # Step 4: Second gated fusion (fusion1 result + local)
        gate2_input = torch.cat([fusion1_out, local_out], dim=-1)
        gate2_weights = self.gate2(gate2_input)
        gate2_out = gate2_weights * local_out + (1 - gate2_weights) * fusion1_out
        
        # Step 5: Optional refinement with strong residual connection
        if self.use_ffn_refinement:
            refinement_norm = self.norm_refinement(gate2_out)
            
            if self.refinement_type == "conv_refinement":
                # For conv refinement, need to transpose
                refinement_input = refinement_norm.transpose(1, 2)
                refined_features = self.refinement(refinement_input).transpose(1, 2)
            else:
                refined_features = self.refinement(refinement_norm)
            
            # CRITICAL: Strong residual connection to preserve gated information
            final_out = gate2_out + self.drop_path_3(refined_features)
        else:
            # No refinement - just use gated output
            final_out = gate2_out
        
        # Use base class method for output extraction
        anchor_out, seq_out = self._extract_anchor_and_sequence_outputs(final_out, anchor_positions, L)
        
        return anchor_out, seq_out, anchor_mask, mask

class AnchorPooling(nn.Module):
    """
    Configurable pooling module for extracting anchor tokens from sequence blocks.
    
    Supports multiple pooling strategies for anchor generation:
    - 'mean': Average pooling over sequence blocks (default, fastest)
    - 'max': Max pooling over sequence blocks (good for sparse features)
    - 'attn': Attention pooling with learnable query (CLIP-style, most expressive)
    - 'gated': Gated pooling that adaptively combines mean and max pooling
    
    Args:
        stride: The stride used for downsampling (tokens per anchor block)
        method: Pooling method to use ('mean', 'max', 'attn', or 'gated')
        d_model: Embedding dimension (required for 'attn' and 'gated' methods)
        nhead: Number of attention heads (used for 'attn' method, default: 2)
        dropout: Dropout probability for attention (default: 0.0)
    
    Input:
        x_blk: Tensor of shape (B, D, num_blocks, stride) - sequence blocks
        
    Output:
        anchors: Tensor of shape (B, D, num_blocks) - pooled anchor tokens
    """
    def __init__(self,
                 stride: int,
                 method: str = "mean",
                 d_model: int = 0,
                 nhead: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.stride = stride
        self.method = method

        if method == "mean":
            self.pooler = MeanPooling(stride)
        elif method == "max":
            self.pooler = MaxPooling(stride)
        elif method == "attn":
            if d_model <= 0:
                raise ValueError("d_model and nhead must be provided for attention pooling")
            self.pooler = AttnPooling(stride, d_model, nhead, dropout)
        elif method == "gated":
            if d_model <= 0:
                raise ValueError("d_model must be provided for gated pooling")
            self.pooler = GatedPooling(stride, d_model)
        else:
            raise ValueError(f"Unknown pooling method {method!r}. Supported methods: 'mean', 'max', 'attn'")

    def forward(self, x_blk: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for anchor pooling.
        
        Args:
            x_blk: Input tensor of shape (B, D, num_blocks, stride)
            
        Returns:
            Pooled anchor tokens of shape (B, D, num_blocks)
        """
        return self.pooler(x_blk)

class MeanPooling(nn.Module):
    """Mean pooling along the last dimension"""
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride
    def forward(self, x_blk: torch.Tensor) -> torch.Tensor:
        # x_blk: (B, D, num_blocks, stride)
        # return x_blk.mean(dim=-1)  # → (B, D, num_blocks)
        return F.avg_pool1d(x_blk, kernel_size=self.stride, stride=self.stride)
        

class MaxPooling(nn.Module):
    """Max pooling along the last dimension"""
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride
    def forward(self, x_blk: torch.Tensor) -> torch.Tensor:
        # return x_blk.max(dim=-1).values
        return F.max_pool1d(x_blk, kernel_size=self.stride, stride=self.stride)
    
class GatedPooling(nn.Module):
    """
    Gated pooling that adaptively combines mean and max pooling operations.
    
    This module learns channel-wise gates to dynamically blend mean pooling
    and max pooling based on the input features. This allows the model to
    adaptively choose between averaging (for distributed features) and 
    max pooling (for sparse/salient features) on a per-channel basis.
    
    Args:
        d_model: Model embedding dimension
        
    Input:
        x: Tensor of shape (B, D, N, S) - N windows of length S
        
    Output:
        pooled: Tensor of shape (B, D, N) - gated pooled features
    """
    def __init__(self, stride: int, d_model: int):
        super().__init__()
        # Conv1d on the channel dimension: (2D) -> D
        self.gate_proj = nn.Conv1d(
            in_channels=2 * d_model,
            out_channels=d_model,
            kernel_size=1,
            groups=1,           # depth‑wise would be groups=d_model
            bias=True
        )
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F.max_pool1d(x, kernel_size=self.stride, stride=self.stride)
        μ   = F.avg_pool1d(x, kernel_size=self.stride, stride=self.stride) # (B, D, N)
        m   = F.max_pool1d(x, kernel_size=self.stride, stride=self.stride) # (B, D, N)
        cat = torch.cat([μ, m], dim=1)  # (B, 2D, N)
        g   = torch.sigmoid(self.gate_proj(cat))  # (B, D, N)
        return g * m + (1.0 - g) * μ

class AttnPooling(nn.Module):
    """
    CLIP-style attention pooling using a learnable query vector.
    
    This module implements attention pooling similar to the approach used in CLIP,
    where a single learnable query vector attends to all tokens in a sequence block
    to produce a pooled representation. This provides more expressive pooling
    compared to simple mean or max pooling.
    
    Args:
        d_model: Model embedding dimension
        nhead: Number of attention heads
        dropout: Dropout probability for attention weights
        
    Input:
        x_blk: Tensor of shape (B, D, num_blocks, stride) - sequence blocks
        
    Output:
        pooled: Tensor of shape (B, D, num_blocks) - attention-pooled features
    """
    def __init__(self, stride: int, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.pool_q = nn.Parameter(torch.randn(1, 1, d_model) / (d_model ** 0.5))  # Initialize with proper scaling
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False
        )
        self.stride = stride

    def forward(self, x_blk: torch.Tensor) -> torch.Tensor:
        B, D, L2 = x_blk.shape
        num_blocks = L2 // self.stride  # Adjust for stride
        x_blk = x_blk.reshape(B, D, num_blocks, self.stride)
        kv = (
            x_blk
            .permute(3, 0, 2, 1)             # (stride, B, num_blocks, D)
            .reshape(self.stride, B * num_blocks, D)
        )
        q = self.pool_q.expand(-1, B * num_blocks, -1)  # (1, B*num_blocks, D)
        out, _ = self.pool_attn(q, kv, kv)               # (1, B*num_blocks, D)
        return (
            out
            .squeeze(0)                # (B*num_blocks, D)
            .view(B, num_blocks, D)    # (B, num_blocks, D)
            .permute(0, 2, 1)          # (B, D, num_blocks)
        )

def anchor_pooling(x: torch.Tensor, stride: int, kernel_size: Optional[int] = None, method: str = "mean") -> torch.Tensor:
    """
    Legacy pooling function - consider using AnchorPooling class instead.
    
    Args:
        x: Input tensor of shape (B, D, L)
        stride: Stride for pooling
        kernel_size: Size of the pooling kernel (defaults to stride)
        method: Pooling method ('mean' or 'max')
        
    Returns:
        Pooled tensor of shape (B, D, ceil(L/stride))
    """
    if kernel_size is None:
        kernel_size = stride
    if method == "mean":
        return F.avg_pool1d(x, kernel_size=kernel_size, stride=stride, ceil_mode=True)
    elif method == "max":
        return F.max_pool1d(x, kernel_size=kernel_size, stride=stride, ceil_mode=True)
    else:
        raise ValueError(f"Unsupported pooling method: {method}. Use 'mean' or 'max'.")


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LayerScale2(nn.Module):
    """
    Multiple residual by a per-channel scaling factor (and zero init) before adding.
    https://arxiv.org/abs/2103.17239
    
    Args:
        n_channels: Number of channels/features
        pdrop: Dropout probability
        init_scale: Initial scale factor
    """
    def __init__(self, n_channels: int, pdrop: float = 0.1, init_scale: float = 1e-4):
        super().__init__()
        # Shape (1, 1, n_channels) for (batch, seq_len, channels) tensor format
        self.scale = nn.Parameter(init_scale * torch.ones((1, 1, n_channels)))
        self.pdrop = pdrop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer scaling and stochastic depth"""
        return drop_path(self.scale.to(x.dtype) * x, self.pdrop, self.training)
    
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Stochastic Depth per sample.
    
    Args:
        x: Input tensor
        drop_prob: Probability of dropping a path
        training: Whether in training mode
        
    Returns:
        Output after applying stochastic depth
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    x = x.div(keep_prob) * mask.floor_()
    return x

def downsample_mask(mask: torch.Tensor, stride: int = 2) -> torch.Tensor:
    """
    Downsample a boolean mask using max pooling with ceiling rounding.
    
    Args:
        mask: Boolean mask of shape (B, 1, L)
        stride: Downsampling factor
        
    Returns:
        Downsampled mask of shape (B, 1, ceil(L/stride))
    """
    # Convert boolean mask to float (0.0 and 1.0)
    mask_float = mask.float()
    
    # Apply max pooling with ceil_mode=True to handle odd lengths
    downsampled_mask_float = F.max_pool1d(
        mask_float, 
        kernel_size=stride, 
        stride=stride, 
        ceil_mode=True  # This handles the ceiling rounding
    )
    
    # Convert back to boolean
    downsampled_mask = downsampled_mask_float.bool()
    
    return downsampled_mask

class EnhancedTransformerEncoder(nn.Module):
    """
    Enhanced transformer encoder with rotary position embeddings.
    
    This class extends the base TransformerEncoder from blocks.py by adding
    rotary position embeddings, which improves modeling of relative positions
    without requiring absolute position embeddings.
    """
    def __init__(
        self, 
        d_model: int, 
        stride: int = 1, 
        window_size: int = 5, 
        n_heads: int = 2,
        use_rotary: bool = True,
        max_position_embeddings: int = 4096,
        expansion: int = 4,
        attn_pdrop: float = 0.0,
        proj_pdrop: float = 0.0,
        path_pdrop: float = 0.0,
    ):
        super().__init__()
        self.use_rotary = use_rotary
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Initialize the base transformer encoder from blocks.py
        self.base_encoder = TransformerEncoder(
            embd_dim=d_model, 
            stride=stride, 
            window_size=window_size, 
            n_heads=n_heads,
            expansion=expansion,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop
        )
        
        # Add rotary embeddings if enabled
        if use_rotary:
            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=max_position_embeddings
            )
            
            # Save original MaskedMHA forward method
            self.original_mha_forward = self.base_encoder.attn.attn.forward
            
            # Replace with our custom forward method that applies rotary embeddings
            self.base_encoder.attn.attn.forward = self._rotary_mha_forward
        
    def _rotary_mha_forward(self, q, k=None, v=None, kv_mask=None, kv_size=None):
        """
        Custom MaskedMHA forward method that applies rotary position embeddings.
        This wraps the original MaskedMHA forward method to apply rotary embeddings
        to the query and key tensors.
        """
        bs, c = q.size(0), self.d_model
        h, d = self.n_heads, self.head_dim
        
        if k is None:
            k = q
        if v is None:
            v = k
        
        # If mask is not given, assume all positions are valid
        if kv_mask is None:
            kv_mask = torch.ones_like(k[:, :1], dtype=torch.bool)
            
        # Apply original linear projections from MaskedMHA
        q_proj = self.base_encoder.attn.attn.query(q)
        k_proj = self.base_encoder.attn.attn.key(k)
        v_proj = self.base_encoder.attn.attn.value(v)
        
        # Extract sequence length
        seq_len = q_proj.size(-1)
        
        # Reshape for attention heads
        q_reshaped = q_proj.view(bs, h, d, -1).permute(0, 1, 3, 2)  # (bs, h, seq_len, d)
        k_reshaped = k_proj.view(bs, h, d, -1).permute(0, 1, 3, 2)  # (bs, h, seq_len, d)
        
        # Apply rotary embeddings
        q_rotary, k_rotary = self.rotary_emb(q_reshaped, k_reshaped, seq_len)
        
        # Reshape back to original format
        q_proj = q_rotary.permute(0, 1, 3, 2).reshape(bs, c, -1)  # (bs, c, seq_len)
        k_proj = k_rotary.permute(0, 1, 3, 2).reshape(bs, c, -1)  # (bs, c, seq_len)
        
        # Continue with the rest of the original MaskedMHA forward
        # We need to implement the attention mechanism directly to avoid infinite recursion
        
        # Get scale and window_size from the original MHA
        scale = self.base_encoder.attn.attn.scale
        window_size = self.base_encoder.attn.attn.window_size
        
        # repeat query to match the size of key / value
        if kv_size is not None and k.size(0) != bs:
            q_proj = q_proj.repeat_interleave(kv_size, dim=0)
            bs = q_proj.size(0)
        
        if window_size > 0:
            # For local attention with window size > 0
            q_flat = q_proj.view(bs, h, d, -1).flatten(0, 1).transpose(1, 2)
            k_flat = k_proj.view(bs, h, d, -1).flatten(0, 1).transpose(1, 2)
            v_flat = v_proj.view(bs, h, d, -1).flatten(0, 1).transpose(1, 2)
            
            # Use original MaskedMHA attention calculation methods
            attn = self.base_encoder.attn.attn._query_key_matmul(q_flat * scale, k_flat * scale)
            attn = attn.view(bs, h, -1, window_size)
            
            # Make sure kv_mask is properly formatted
            kv_mask_t = kv_mask.transpose(1, 2)
            attn = self.base_encoder.attn.attn._attn_normalize(attn, kv_mask_t)
            attn = self.base_encoder.attn.attn.attn_drop(attn)
            attn = attn.view(bs * h, -1, window_size)
            q_out = self.base_encoder.attn.attn._attn_value_matmul(attn, v_flat)
            q_out = q_out.view(bs, h, -1, d)
        else:
            # For global attention
            q_out = q_proj.view(bs, h, d, -1).transpose(2, 3)
            k_out = k_proj.view(bs, h, d, -1)
            v_out = v_proj.view(bs, h, d, -1).transpose(2, 3)
            
            # Compute attention scores
            attn = torch.matmul(q_out * scale, k_out * scale)
            
            # Apply mask
            attn = attn.masked_fill(
                mask=torch.logical_not(kv_mask[:, :, None, :]),
                value=float('-inf'),
            )
            
            # Apply softmax and dropout
            attn = F.softmax(attn, dim=-1)
            attn = self.base_encoder.attn.attn.attn_drop(attn)
            
            # Apply attention to values
            q_out = torch.matmul(attn, v_out)
        
        # Final projection and dropout
        q_out = q_out.transpose(2, 3).reshape(bs, c, -1)
        out = self.base_encoder.attn.attn.proj_drop(self.base_encoder.attn.attn.proj(q_out))
        
        return out
    
    def forward(self, x, mask=None):
        """
        Forward pass with rotary embeddings.
        
        Args:
            x: Input tensor of shape (B, D, L)
            mask: Optional boolean mask of shape (B, 1, L)
            
        Returns:
            Tuple of (output tensor, mask)
        """
        # Simply use the base encoder's forward method
        # Rotary embeddings are applied within the attention mechanism
        return self.base_encoder(x, mask)
        
    def __del__(self):
        # Restore original forward method if this object is deleted
        if self.use_rotary and hasattr(self, 'original_mha_forward'):
            if hasattr(self.base_encoder, 'attn') and hasattr(self.base_encoder.attn, 'attn'):
                self.base_encoder.attn.attn.forward = self.original_mha_forward

class RotaryEmbedding(nn.Module):
    """
    Rotary position embeddings module.
    Based on: https://arxiv.org/abs/2104.09864
    
    This technique replaces absolute positional embeddings with relative position-dependent
    transformations to capture relative position information through rotation.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 4096):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        
        # Initialize frequencies for rotation
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for sin/cos values based on position
        self._rotary_cache = {}
        
    def _get_rotary_cache(self, seq_len, device):
        """Generate or retrieve cached rotation matrices for given sequence length"""
        cache_key = (seq_len, str(device))
        if cache_key not in self._rotary_cache:
            # Generate position indices
            positions = torch.arange(seq_len, device=device)
            
            # Calculate sin/cos values
            freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            
            # Cache results
            self._rotary_cache[cache_key] = (cos, sin)
            
        return self._rotary_cache[cache_key]
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims of x"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q, k, seq_len=None):
        """
        Apply rotary position embeddings to query and key tensors
        
        Args:
            q: Query tensor of shape (..., seq_len, dim)
            k: Key tensor of shape (..., seq_len, dim)
            seq_len: Optional sequence length if different from tensor dimension
            
        Returns:
            Tuple of rotated (q, k) tensors
        """
        seq_len = seq_len or q.shape[-2]
        cos, sin = self._get_rotary_cache(seq_len, q.device)
        
        # Reshape for broadcasting
        cos = cos[:seq_len].unsqueeze(0)  # (1, seq_len, dim)
        sin = sin[:seq_len].unsqueeze(0)  # (1, seq_len, dim)
        
        # Apply rotary embeddings
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed


class MambaWindowEncoder(nn.Module):
    """
    Mamba-based window encoder as an alternative to TransformerEncoder.
    Uses Hydra for efficient sequence modeling within local windows.
    """
    def __init__(
        self,
        embd_dim,           # embedding dimension
        stride=1,           # convolution stride (0 if disable convs)
        window_size=0,      # window size for local processing
        n_heads=4,          # number of attention heads (equivalent parameter for Hydra)
        expansion=4,        # expansion factor for FFN
        d_state=64,         # state dimension for Mamba (SSM)
        d_conv=7,           # convolution kernel size in Hydra
        head_dim=48,        # dimension per head
        dropout=0.0,        # dropout rate
        path_pdrop=0.0,     # dropout rate for residual paths
    ):
        super().__init__()

        self.embd_dim = embd_dim
        self.window_size = window_size
        self.stride = stride
        
        # Optional downsampling layer
        if stride > 1:
            self.downsample = MaskedMaxPool1D(3, stride=stride)
        else:
            self.downsample = None
            
        # Layer normalization for input
        self.norm_in = RMSNorm(embd_dim)
        
        # Hydra for sequence modeling (equivalent to self-attention)
        self.hydra_encoder = Hydra(
            d_model=embd_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expansion // 2,  # Usually Hydra uses smaller expansion
            headdim=head_dim,
            use_mem_eff_path=True
        )
        
        self.drop_path_1 = LayerScale2(embd_dim, path_pdrop)
        
        # FFN layer (similar to TransformerEncoder)
        self.ffn = nn.Sequential(
            nn.Linear(embd_dim, expansion * embd_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * embd_dim, embd_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = RMSNorm(embd_dim)
        self.drop_path_2 = LayerScale2(embd_dim, path_pdrop)

    def _process_windows(self, x, mask=None):
        """
        Process input sequence in sliding windows with memory-efficient implementation
        
        Args:
            x: Input tensor of shape (B, L, D)
            mask: Mask tensor of shape (B, 1, L) or None
            
        Returns:
            Output tensor of same shape as x
        """
        B, L, D = x.shape
        
        if self.window_size <= 0 or L <= self.window_size:
            # If window size is 0 or sequence is shorter than window, process whole sequence
            return self.hydra_encoder(x)
        
        # Process in sliding windows with overlap
        stride = self.window_size // 2
        
        # Pad input to ensure all positions are processed
        padded_len = ((L - 1) // stride + 1) * stride + self.window_size
        padding = padded_len - L
        
        if padding > 0:
            x_padded = F.pad(x, (0, 0, 0, padding))
            if mask is not None:
                mask_padded = F.pad(mask, (0, padding))
            else:
                mask_padded = None
        else:
            x_padded = x
            mask_padded = mask
            
        # Calculate number of windows
        num_windows = (padded_len - self.window_size) // stride + 1
        
        # Create linear weights for blending (higher in middle, lower at edges)
        if self.window_size > 1:
            window_weights = torch.linspace(0, 1, self.window_size//2 + 1, device=x.device)
            window_weights = torch.cat([
                window_weights,
                torch.flip(window_weights[:-1], [0])
            ]).view(1, self.window_size, 1)
        else:
            window_weights = torch.ones(1, self.window_size, 1, device=x.device)
        
        # Initialize output and weight accumulator tensors
        output = torch.zeros(B, padded_len, D, device=x.device)
        weight_acc = torch.zeros(B, padded_len, 1, device=x.device)
        
        # Dynamically adjust batch size based on input batch size and available memory
        # If B=1, we can process more windows in each batch
        if B == 1:
            batch_size = max(1, 4096 // self.window_size)  # Larger batch for B=1
        else:
            batch_size = max(1, 2048 // self.window_size)
            
        # Process windows in batches to save memory
        for start_idx in range(0, num_windows, batch_size):
            end_idx = min(start_idx + batch_size, num_windows)
            current_batch_size = end_idx - start_idx
            
            # More efficient window collection - for B=1 case, use unfold directly
            if B == 1 and current_batch_size > 1:
                # Get all window start positions for this batch
                window_starts = torch.arange(start_idx, end_idx, device=x.device) * stride
                
                # For B=1, we can create a 3D tensor of all windows directly
                windows_batch = []
                for i in range(current_batch_size):
                    window_start = start_idx * stride + i * stride
                    window_end = window_start + self.window_size
                    windows_batch.append(x_padded[:, window_start:window_end])
                windows_batch = torch.cat(windows_batch, dim=0)  # (current_batch_size, window_size, D)
                
                # Reshape for batch processing
                windows_batch = windows_batch.unsqueeze(0)  # Add batch dim: (1, current_batch_size, window_size, D)
                
                # Apply masks if needed
                if mask_padded is not None:
                    mask_batch = []
                    for i in range(current_batch_size):
                        window_start = start_idx * stride + i * stride
                        window_end = window_start + self.window_size
                        mask_batch.append(mask_padded[:, :, window_start:window_end])
                    mask_batch = torch.cat(mask_batch, dim=0).transpose(0, 1)  # (1, current_batch_size, window_size)
                    mask_batch = mask_batch.unsqueeze(-1)  # (1, current_batch_size, window_size, 1)
                    windows_batch = windows_batch * mask_batch.to(windows_batch.dtype)
            else:
                # Standard collection method for B > 1
                # Collect windows for current batch
                windows_batch = []
                window_positions = []
                
                for i in range(start_idx, end_idx):
                    window_start = i * stride
                    window_end = window_start + self.window_size
                    windows_batch.append(x_padded[:, window_start:window_end])
                    window_positions.append(window_start)
                
                # Stack windows into a single batch
                windows_batch = torch.stack(windows_batch, dim=1)  # (B, current_batch_size, window_size, D)
                
                # Apply masks if needed
                if mask_padded is not None:
                    mask_batch = []
                    for i in range(start_idx, end_idx):
                        window_start = i * stride
                        window_end = window_start + self.window_size
                        mask_batch.append(mask_padded[:, :, window_start:window_end])
                    
                    mask_batch = torch.stack(mask_batch, dim=1)  # (B, current_batch_size, 1, window_size)
                    mask_batch = mask_batch.squeeze(2).unsqueeze(-1)  # (B, current_batch_size, window_size, 1)
                    windows_batch = windows_batch * mask_batch.to(windows_batch.dtype)
            
            # Reshape for Hydra processing
            batch_size_for_hydra = B * current_batch_size
            windows_flat = windows_batch.reshape(batch_size_for_hydra, self.window_size, D)
            
            # Process with Hydra
            processed_windows = self.hydra_encoder(windows_flat)
            
            # Reshape back
            processed_windows = processed_windows.reshape(B, current_batch_size, self.window_size, D)
            
            # Create positions for current window batch
            window_positions = torch.arange(start_idx, end_idx, device=x.device) * stride
            
            # Apply window weights and accumulate - vectorized for the whole batch when possible
            for i in range(current_batch_size):
                window_start = window_positions[i].item()
                window_end = window_start + self.window_size
                
                # Apply weights to current window
                weighted_window = processed_windows[:, i] * window_weights
                
                # Add to output and accumulate weights
                output[:, window_start:window_end] += weighted_window
                weight_acc[:, window_start:window_end] += window_weights
        
        # Normalize by accumulated weights
        output = output / (weight_acc + 1e-6)
        
        # Return just the needed part
        return output[:, :L]

    def forward(self, x, mask=None):
        """
        Forward pass of the Mamba window encoder
        
        Args:
            x: Input tensor of shape (B, D, L)
            mask: Mask tensor of shape (B, 1, L) or None
            
        Returns:
            Tuple of (output tensor, mask)
        """
        if mask is None:
            mask = torch.ones((x.shape[0], 1, x.shape[2]), dtype=torch.bool, device=x.device)
            
        # Apply input mask
        x = x * mask.to(x.dtype)
        
        # Optional downsampling
        if self.downsample is not None:
            x_skip, mask = self.downsample(x, mask)
        else:
            x_skip = x
            
        # Transpose to (B, L, D) for Hydra
        x_t = x.transpose(1, 2)
        
        # Apply layer norm
        x_norm = self.norm_in(x_t)
        
        # Process with Hydra in windows
        h = self._process_windows(x_norm, mask)
        
        # Residual connection
        x_t = x_t + self.drop_path_1(h)
        
        # Apply FFN
        x_for_ffn = self.norm_ffn(x_t)
        
        # Process FFN in chunks to save memory if sequence is long
        if x_for_ffn.size(1) > 2048:
            chunk_size = 2048
            h_chunks = []
            
            for i in range(0, x_for_ffn.size(1), chunk_size):
                end_idx = min(i + chunk_size, x_for_ffn.size(1))
                h_chunk = self.ffn(x_for_ffn[:, i:end_idx])
                h_chunks.append(h_chunk)
                
            h = torch.cat(h_chunks, dim=1)
        else:
            h = self.ffn(x_for_ffn)
            
        x_t = x_t + self.drop_path_2(h)
        
        # Transpose back to (B, D, L)
        x_out = x_t.transpose(1, 2)
        
        # Apply output mask
        x_out = x_out * mask.to(x_out.dtype)
        
        return x_out, mask

class LocalDWConvGLU(nn.Module):
    def __init__(self, d_model, k=5, expansion=2, pdrop=0.1):
        super().__init__()
        self.dw = nn.Conv1d(d_model, d_model, k,
                            padding=k//2, groups=d_model, bias=False)
        self.pw1 = nn.Linear(d_model, expansion * d_model * 2)   # GLU
        self.pw2 = nn.Linear(expansion * d_model, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(pdrop)

    def forward(self, x, mask):
        x = x * mask.to(x.dtype)          # (B, D, L)
        h = self.dw(x)                    # depth-wise local mixing
        h = h.transpose(1, 2)             # (B, L, D)
        v, g = self.pw1(h).chunk(2, dim=-1)
        h = self.act(v) * torch.sigmoid(g)   # gated channel mixing
        h = self.pw2(h).transpose(1, 2)
        return (h + x) * mask.to(x.dtype), mask
