from dataclasses import dataclass

import torch
import torch.nn as nn
import torchtune
from huggingface_hub import PyTorchModelHubMixin
from torchtune.models import llama3_2


def llama3_2_1B() -> torchtune.modules.transformer.TransformerDecoder:
    """Create a 1B parameter Llama 3.2 model configuration."""
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_100M() -> torchtune.modules.transformer.TransformerDecoder:
    """Create a 100M parameter Llama 3.2 model configuration."""
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )

# TODO: Change so dims are same but layers are smaller or some shit
def llama3_2_downsized() -> torchtune.modules.transformer.TransformerDecoder:
    """Create a downsized Llama 3.2 model configuration."""
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=2, # 1/2
        num_heads=4, # 1/2
        num_kv_heads=2,
        embed_dim=1024, # keep same as llama3_2_100M
        max_seq_len=2048,
        intermediate_dim=4096, # 1/2 of 8192
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


# Dictionary mapping model flavor names to their constructor functions
FLAVORS = {
    "llama-1B": llama3_2_1B,
    "llama-100M": llama3_2_100M,
    "llama-downsized": llama3_2_downsized,
}


def _prepare_transformer(model):
    """
    Prepare transformer model by replacing token embeddings and output layers with Identity.
    
    Returns:
        tuple: (modified model, embedding dimension)
    """
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()  # Replace with Identity since we'll handle embeddings separately
    model.output = nn.Identity()  # Replace output projection with Identity
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    """
    Create a causal attention mask (lower triangular) of shape (seq_len, seq_len).
    
    Returns:
        torch.Tensor: Boolean tensor of shape (seq_len, seq_len)
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    Index into a causal mask using position indices.
    
    Args:
        mask: (max_seq_len, max_seq_len) - Full causal mask
        input_pos: (batch_size, seq_len) - Position indices for each token
    
    Returns:
        torch.Tensor: (batch_size, seq_len, max_seq_len) - Indexed causal mask
    """
    r = mask[input_pos, :]  # Index into mask using input positions
    return r


def _multinomial_sample_one_no_sync(probs):
    """
    Perform multinomial sampling without CUDA synchronization for efficiency.
    
    Args:
        probs: (..., vocab_size) - Probability distribution
        
    Returns:
        torch.Tensor: (..., 1) - Sampled token indices
    """
    q = torch.empty_like(probs).exponential_(1)  # Sample from exponential distribution
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)  # Gumbel-max trick


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    """
    Sample from the top-k logits with temperature scaling.
    
    Args:
        logits: (..., vocab_size) - Unnormalized logits
        topk: int - Number of top candidates to sample from
        temperature: float - Controls randomness (higher = more random)
        
    Returns:
        torch.Tensor: (..., 1) - Sampled token indices
    """
    # Apply temperature scaling
    logits = logits / temperature  # (..., vocab_size)

    # Filter to keep only top-k logits
    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]  # (..., vocab_size)
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)  # (..., vocab_size)
    
    # Convert to probabilities
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)  # (..., vocab_size)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)  # (..., vocab_size)

    # Sample from the probability distribution
    sample_token = _multinomial_sample_one_no_sync(probs)  # (..., 1)
    return sample_token


@dataclass
class ModelArgs:
    """Configuration for the CSM model."""
    backbone_flavor: str  # Name of the backbone transformer model
    decoder_flavor: str   # Name of the decoder transformer model
    text_vocab_size: int  # Size of the text vocabulary
    audio_vocab_size: int  # Size of the audio vocabulary per codebook
    audio_num_codebooks: int  # Number of audio codebooks


class Model(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/SesameAILabs/csm",
    pipeline_tag="text-to-speech",
    license="apache-2.0",
):
    def __init__(self, config: ModelArgs):
        """
        Initialize the CSM model with backbone and decoder transformers.
        """
        super().__init__()
        self.config = config

        # Initialize backbone and decoder transformers
        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[config.backbone_flavor]())
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[config.decoder_flavor]())
        self.draft_decoder, draft_decoder_dim = _prepare_transformer(FLAVORS["llama-downsized"]()) # also untrained rn bruh

        print(f"Backbone dimension: {backbone_dim}")
        print(f"Decoder dimension: {decoder_dim}")
        print(f"Draft decoder dimension: {draft_decoder_dim}")

        # Embedding layers for text and audio tokens
        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)  # (text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.audio_num_codebooks, backbone_dim
        )  # (audio_vocab_size * audio_num_codebooks, backbone_dim)

        # Projection from backbone dimension to decoder dimension
        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)  # (backbone_dim, decoder_dim)
        
        # Head for generating the first codebook tokens
        self.codebook0_head = nn.Linear(backbone_dim, config.audio_vocab_size, bias=False)  # (backbone_dim, audio_vocab_size)
        
        # Heads for generating subsequent codebook tokens
        # Shape: (audio_num_codebooks-1, decoder_dim, audio_vocab_size)
        # This is a stack of linear projection matrices, one for each codebook after the first one. Each matrix:
        # Takes decoder hidden states (decoder_dim)
        # project them to logits over the audio vocabulary (audio_vocab_size)
        self.audio_head = nn.Parameter(
            torch.empty(config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size)
        )

    def setup_caches(self, max_batch_size: int) -> torch.Tensor:
        """
        Setup KV caches for efficient autoregressive generation.
        
        Args:
            max_batch_size: int - Maximum batch size for generation
            
        Returns:
            torch.Tensor: Causal mask for attention
        """
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        with device:
            # Setup caches for both backbone and decoder transformers
            self.backbone.setup_caches(max_batch_size, dtype)
            self.decoder.setup_caches(max_batch_size, dtype, decoder_max_seq_len=self.config.audio_num_codebooks)
            self.draft_decoder.setup_caches(max_batch_size, dtype, decoder_max_seq_len=self.config.audio_num_codebooks)
            
        # Create and register causal masks as buffers
        self.register_buffer("backbone_causal_mask", _create_causal_mask(self.backbone.max_seq_len, device))  # (max_seq_len, max_seq_len)
        self.register_buffer("decoder_causal_mask", _create_causal_mask(self.config.audio_num_codebooks, device))  # (audio_num_codebooks, audio_num_codebooks)

    def generate_frame_speculative(
        self, 
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int) -> torch.Tensor:
        """
        Performs speculative decoding on the decoder using the draft model.
        """
        # (Step 0) generate c0 with codebook0_head as before

        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size() 
        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2) 
        
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)
        last_h = h[:, -1, :]
        
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample)

        # Initialize current hidden state and sampled tokens
        # Shape of curr_h: (batch_size, 2, backbone_dim) - concatenating last_h and c0_embed
        # Shape of curr_sample: (batch_size, 1)
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1) # combine the last hidden state from the backbone and the first codebook embedding
        curr_sample = c0_sample.clone()
        
        # Create position indices for decoder
        # Shape: (batch_size, 2) 
        # literally always exactly [0, 1] bc only 2 elements in sequence
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

        # Reset KV caches
        self.decoder.reset_caches()
        self.draft_decoder.reset_caches()

        # Speculative decoding loop
        while curr_sample.shape[1] < self.config.audio_num_codebooks:

            # Guessing 3 codebooks ahead at a time.
            speculative_len = min(3, self.config.audio_num_codebooks - curr_sample.shape[1])  # Tune this
            speculative_samples = []

            # Step 1: Generate speculative samples using the draft decoder
            for _ in range(speculative_len):
                # Create causal mask for current positions
                mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
                
                # Run the draft decoder on current hidden states
                draft_h: torch.Tensor = self.draft_decoder(self.projection(curr_h), input_pos=curr_pos, mask=mask)
                
                # Get the index of the next codebook to predict
                i = curr_sample.shape[1]
                
                # Take the decoder output for the last position (draft_h[:, -1, :])
                # Multiply it by the specific weight matrix for this codebook (self.audio_head[i - 1])
                # This gives us logits over the vocabulary for this codebook
                # sample a token using topk sampling with temperature
             
                # Convert draft_h to same dtype as audio_head before matmul
                logits = torch.matmul(draft_h[:, -1, :].to(self.audio_head.dtype), self.audio_head[i - 1])
                sample = sample_topk(logits, topk, temperature) # (B, 1)

                speculative_samples.append(sample)

                # Embed the sampled token
                embed = self._embed_audio(i, sample) # (B, 1, D)
                
                # Update current hidden state with the new embedding
                curr_h = embed
                
                # Increment position indices for next token
                curr_pos = curr_pos[:, -1:] + 1
                
                # Append the sampled token to our running sequence
                curr_sample = torch.cat([curr_sample, sample], dim=1)

            # Step 2: Verify speculative samples with the full decoder
            print(f"Speculative samples: {speculative_samples}")
            

            
            # Process all tokens at once
            # full_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, mask=mask)



        # Return the complete sequence of audio tokens
        return curr_sample


    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """
        Generate a single frame of audio tokens autoregressively.
        
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1) - Input tokens including text and audio
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1) - Mask for input tokens
            input_pos: (batch_size, seq_len) - Position indices for each token
            temperature: float - Controls randomness in sampling (higher = more random)
            topk: int - Number of top candidates to sample from
            
        Returns:
            (batch_size, audio_num_codebooks) - Sampled audio tokens for a single frame
        """

        # Get model dtype for consistent computation
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()  # b: batch_size, s: sequence_length

        # Ensure backbone caches are enabled for efficient generation
        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        
        # Get the appropriate causal mask for the current positions
        # Shape: (batch_size, seq_len, max_seq_len)
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        
        # Embed the input tokens
        # Shape: (batch_size, seq_len, audio_num_codebooks+1, backbone_dim)
        embeds = self._embed_tokens(tokens)
        
        # Apply mask to embeddings and sum across the token dimension
        # Shape after mask: (batch_size, seq_len, audio_num_codebooks+1, backbone_dim)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        # ^ This mask is not a causal mask but it is used because there are 2 modalities here: text and audio. When it's a text embedding, the audio positions should be zeroed out and vice versa.

        # Shape after sum: (batch_size, seq_len, backbone_dim)
        h = masked_embeds.sum(dim=2) 
        
        # Process through backbone transformer
        # Shape: (batch_size, seq_len, backbone_dim)
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)

        # Get the last hidden state for generating the first codebook
        # Shape: (batch_size, backbone_dim)
        last_h = h[:, -1, :]
        
        # Generate the first codebook (c0) token
        # Shape of c0_logits: (batch_size, audio_vocab_size)
        c0_logits = self.codebook0_head(last_h)

        # Shape of c0_sample: (batch_size, 1) - just doing it over the batch dim
        c0_sample = sample_topk(c0_logits, topk, temperature)
        
        # Embed the sampled token
        # Shape: (batch_size, 1, backbone_dim)
        c0_embed = self._embed_audio(0, c0_sample)

        # Initialize current hidden state and sampled tokens
        # Shape of curr_h: (batch_size, 2, backbone_dim) - concatenating last_h and c0_embed
        # Shape of curr_sample: (batch_size, 1)
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1) # combine the last hidden state from the backbone and the first codebook embedding
        curr_sample = c0_sample.clone()
        
        # Create position indices for decoder
        # Shape: (batch_size, 2) 
        # literally always exactly [0, 1] bc only 2 elements in sequence
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

        # Reset decoder caches for new frame generation
        self.decoder.reset_caches()
        
        # Autoregressively generate remaining codebooks
        for i in range(1, self.config.audio_num_codebooks):
            # Get causal mask for current decoder positions
            # Shape: (batch_size, curr_pos_len, audio_num_codebooks)
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            
            # Project backbone hidden states to decoder dimension
            # Shape after projection: (batch_size, curr_pos_len, decoder_dim)
            # Shape after decoder: (batch_size, curr_pos_len, decoder_dim)
            decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask).to(
                dtype=dtype
            )
            
            # Generate token for current codebook
            # Shape of decoder_h[:, -1, :]: (batch_size, decoder_dim)
            # Shape of audio_head[i-1]: (decoder_dim, audio_vocab_size)
            # Shape of ci_logits: (batch_size, audio_vocab_size)
            # Shape of ci_sample: (batch_size, 1)
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)

            # EXPLANATION: Now we
            # Take the decoder output for the last position (decoder_h[:, -1, :])
            # Multiply it by the specific weight matrix for this codebook (self.audio_head[i - 1])
            # This gives us logits over the vocabulary for this codebook
            # sample a token using topk sampling with temperature
            
            # Embed the sampled token
            # Shape: (batch_size, 1, backbone_dim)
            ci_embed = self._embed_audio(i, ci_sample)

            # Update current hidden state with new embedding
            # Shape: (batch_size, 1, backbone_dim)
            curr_h = ci_embed
            
            # Append new token to the sample
            # Shape: (batch_size, i+1)
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            
            # Increment position for next token
            # Shape: (batch_size, 1)
            curr_pos = curr_pos[:, -1:] + 1

        # Return the complete frame of audio tokens
        # Shape: (batch_size, audio_num_codebooks)
        return curr_sample

    def reset_caches(self):
        """Reset KV caches for both backbone and decoder transformers."""
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed audio tokens for a specific codebook.
        
        Args:
            codebook: int - Codebook index
            tokens: (batch_size, 1) - Token indices
            
        Returns:
            torch.Tensor: (batch_size, 1, backbone_dim) - Embedded tokens
        """
        # Offset tokens by codebook index * vocab_size to get the correct embedding indices
        # Shape: (batch_size, 1, backbone_dim)
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Converts raw token IDs in the form of (B, S, A+1) into vector embeddings.
        It takes the last column of tokens as text tokens (tokens[:, :, -1])
        It takes all other columns as audio tokens (tokens[:, :, :-1])
        It embeds each separately with different embedding tables
        It concatenates them back together
        
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1) - Input tokens
                   Last dimension contains audio tokens (:-1) and text token (-1)
            
        Returns:
            torch.Tensor: (batch_size, seq_len, audio_num_codebooks+1, backbone_dim) - Embedded tokens
        """
        # Embed text tokens (last column of tokens tensor)
        # Shape: (batch_size, seq_len, 1, backbone_dim)
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)

        # EXPLANATION:
        # unsqueeze(-2) adds a dimension, making shape (batch_size, seq_len, 1, backbone_dim)
        # The added dimension is necessary for the later concatenation with audio embeddings.

        # Embed audio tokens (all columns except the last one)
        # Calculate indices for audio embeddings by adding offset based on codebook
        # Shape of audio_tokens: (batch_size, seq_len, audio_num_codebooks)
        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        # EXPLANATION:
        # Is performing an "index shift" for each codebook. Let me break down what this addition actually does:
        # arange gives: tensor [0, 1, 2, ..., num_codebooks-1]
        # -> [0, vocab_size, 2*vocab_size, ...]
        # -> [0, 1024, 2048, 3072, ..., 31*1024]
        # When added to the audio tokens, it shifts the index of the audio tokens by the amount specified by the codebook index.
        # For codebook 0: original token IDs (0 to 1023)
        # For codebook 1: original token IDs + 1024 (1024 to 2047)
        # For codebook 2: original token IDs + 2048 (2048 to 3071)

        # EXAMPLE:
        # Original tokens: [[5, 2]]  # Token 5 from codebook 0, token 2 from codebook 1

        # vocab_size = 4
        # codebook_offsets = [0, 4]  # 0*4, 1*4

        # After addition:
        # [[5, 6]]  # 5+0, 2+4
        
        # Reshape all the token ids from (B, S, A) into a single 1D list. Apply embeddings to all of them and then reshape back to (B, S, A, D) where D is the backbone dimension.
        # Shape after view: (batch_size*seq_len*audio_num_codebooks)
        # Shape after embedding: (batch_size*seq_len*audio_num_codebooks, backbone_dim)
        # Shape after reshape: (batch_size, seq_len, audio_num_codebooks, backbone_dim)
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )

        # Concatenate audio and text embeddings along the token dimension
        # Shape: (batch_size, seq_len, audio_num_codebooks+1, backbone_dim)

        # EXPLANATION:
        # audio_embeds: [B, S, C, D]       text_embeds: [B, S, 1, D]
        #     │                               │
        #     │                               │
        #     └─────────────┬─────────────────┘
        #                   │
        #                   v
        # Combined: [B, S, C+1, D]

        # Where:

        # B = batch size
        # S = sequence length
        # C = number of codebooks
        # D = embedding dimension
        return torch.cat([audio_embeds, text_embeds], dim=-2)

if __name__ == "__main__":
    # Create a Llama 10M model to check parameters
    # llama_100m = llama3_2_100M()
    # llama_10m = llama3_2_downsized()
    # num_params = sum(p.numel() for p in llama_100m.parameters())
    # print(f"Llama 100M model has {num_params:,} parameters")

    # num_params = sum(p.numel() for p in llama_10m.parameters())
    # print(f"Llama 10M model has {num_params:,} parameters")
    
    # # Create the full model
    model = Model.from_pretrained("sesame/csm-1b")
    # model = Model(ModelArgs(backbone_flavor="llama-100M", decoder_flavor="llama-downsized", text_vocab_size=128_256, audio_vocab_size=128_256, audio_num_codebooks=16))
    print(model)
