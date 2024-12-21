import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, TaskType
from typing import Optional, Dict, Any, Tuple, Union

from rl4co.utils.pylogger import get_pylogger
from torch import Tensor

from routefinder.models.env_embeddings.mtvrp import MTVRPInitEmbeddingRouteFinder
from routefinder.models.nn.transformer import Normalization, TransformerBlock

log = get_pylogger(__name__)

class BatchProcessingCombiner(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.projection = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, init_h, llama_embeddings):
        # Get shapes
        B_init, N, H = init_h.shape
        B_llama, S, _ = llama_embeddings.shape
        
        # Handle batch size mismatch by processing in chunks
        chunk_size = min(B_init, B_llama)
        combined_chunks = []
        
        for i in range(0, B_init, chunk_size):
            # Get current chunk of init_h
            end_idx = min(i + chunk_size, B_init)
            init_chunk = init_h[i:end_idx]
            
            # Get or repeat llama embeddings as needed
            if B_llama == B_init:
                llama_chunk = llama_embeddings[i:end_idx]
            else:
                # Repeat or slice llama embeddings to match batch size
                llama_chunk = llama_embeddings[:chunk_size]
                if end_idx - i < chunk_size:
                    llama_chunk = llama_chunk[:end_idx-i]
            
            # Mean pool LLM embeddings for this chunk
            llama_pooled = llama_chunk.mean(dim=1, keepdim=True)  # [chunk_size, 1, H]
            
            # Expand to match number of nodes
            llama_expanded = llama_pooled.expand(-1, N, -1)  # [chunk_size, N, H]
            
            # Combine embeddings for this chunk
            chunk_combined = torch.cat([init_chunk, llama_expanded], dim=-1)  # [chunk_size, N, 2H]
            chunk_combined = self.projection(chunk_combined)  # [chunk_size, N, H]
            chunk_combined = self.norm(chunk_combined)
            
            combined_chunks.append(chunk_combined)
        
        # Concatenate all processed chunks
        return torch.cat(combined_chunks, dim=0)  # [B_init, N, H]


class RouteFinderEncoder(nn.Module):
    """
    Encoder for RouteFinder model based on the Transformer Architecture.
    Here we include additional embedding from raw to embedding space, as
    well as more modern architecture options compared to the usual Attention Models
    based on POMO (including multi-task VRP ones).
    """

    def __init__(
        self,
        init_embedding: nn.Module = None,
        num_heads: int = 8,
        embed_dim: int = 128,
        num_layers: int = 6,
        feedforward_hidden: int = 512,
        normalization: str = "instance",
        use_prenorm: bool = False,
        use_post_layers_norm: bool = False,
        parallel_gated_kwargs: dict = None,
        **transformer_kwargs,
    ):
        super(RouteFinderEncoder, self).__init__()

        if init_embedding is None:
            init_embedding = MTVRPInitEmbeddingRouteFinder(embed_dim=embed_dim)
        else:
            log.warning("Using custom init_embedding")
        self.init_embedding = init_embedding
        self.device = torch.device('cuda', torch.cuda.current_device()) #if 'device' not in model_params.keys() else model_params['device']

        # Initialize Llama-2 with optimizations
        self._init_llama()

        # Assuming Llama-2 uses the same or compatible embedding dimension
        # self.llama_embedding_projection = nn.Linear(self.llama_model.config.hidden_size, embed_dim)

        # llama_hidden_size = 896  # Llama-2 7B hidden size
        ff_hidden_dim = 512
        hidden_dim = ff_hidden_dim
        embedding_dim = embed_dim
        
        # Project Llama embeddings to the model's embedding dimension with dtype torch.bfloat16
        self.llm_projection = nn.Sequential(
            nn.Linear(self.llama_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        ).to(dtype=torch.bfloat16)
        
        self.layers = nn.Sequential(
            *(
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    normalization=normalization,
                    use_prenorm=use_prenorm,
                    feedforward_hidden=feedforward_hidden,
                    parallel_gated_kwargs=parallel_gated_kwargs,
                    **transformer_kwargs,
                )
                for _ in range(num_layers)
            )
        )

        self.post_layers_norm = (
            Normalization(embed_dim, normalization) if use_post_layers_norm else None
        )

        # In __init__
        self.embedding_combiner = BatchProcessingCombiner(embed_dim=embed_dim)

    def _init_llama(self):
        """Initialize Llama-2 7B with memory optimizations"""
        # model_name = "//common/public/LLAMA2-HF/Llama-2-7b-chat-hf"  # or local path
        model_name = "Qwen/Qwen2-0.5B"
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', truncation_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization and optimization configs
        model_config = AutoConfig.from_pretrained(model_name)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            bnb_4bit_use_double_quant=True,
        )

        self.llama = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            output_hidden_states=True
        )
        # Freeze Llama parameters
        for param in self.llama.parameters():
            param.requires_grad = False
            
        self.llama_hidden_size = self.llama.config.hidden_size

    def forward(
        self, td: Tensor, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:

        # Transfer to embedding space with initial embedding
        # print(125, td)
        print(174, len(td['prompt']), len(td['locs']), len(td['prompt'][0]), len(td['locs'][0]))
        init_h = self.init_embedding(td)  # [B, N, H]

        # Extract text descriptions for embedding
        if 'prompt' in td.keys():
            # print(131, "forward", td['prompt'].tolist())
            
            with torch.no_grad():  # Don't compute gradients for LLM inference
                tokenized_text = self.tokenizer(td['prompt'].tolist(), return_tensors="pt", padding=True, truncation=True).to(self.device)
                print(183, len(td['prompt'][-1]), len(tokenized_text[-1]))
                llama_outputs = self.llama(**tokenized_text, output_hidden_states=True)
                llama_embeddings = llama_outputs.hidden_states[-1]  # [B, S, LLM_H]
            print(186)
            # Project LLM embeddings to match the dimension of the route finder
            llama_embeddings_projected = self.llm_projection(llama_embeddings)  # [B, S, H]
        print(189)
        # Combine embeddings
        combined_embeddings = self.embedding_combiner(init_h, llama_embeddings_projected)
        print(192)
        # Process through transformer layers
        h = combined_embeddings
        for layer in self.layers:
            h = layer(h, mask)
        
        if self.post_layers_norm is not None:
            h = self.post_layers_norm(h)
        
        return h, init_h


        # # Extract text descriptions for embedding
        # if 'prompt' in td.keys():
        #     # print(131, "forward", td['prompt'].tolist())
            
        #     with torch.no_grad():  # Don't compute gradients for LLM inference
        #         tokenized_text = self.tokenizer(td['prompt'].tolist(), return_tensors="pt", padding=True, truncation=True).to(self.device)
        #         print(138, len(td['prompt'][-1]), len(tokenized_text[-1]))
        #         llama_outputs = self.llama(**tokenized_text, output_hidden_states=True)
        #         llama_embeddings = llama_outputs.hidden_states[-1]  # [B, S, LLM_H]
            
        #     # Project LLM embeddings to match the dimension of the route finder
        #     llama_embeddings_projected = self.llm_projection(llama_embeddings)  # [B, S, H]

        #     print(145, "forward", init_h.shape, llama_embeddings_projected.shape, llama_embeddings.shape)
            
        #     # Combine the embeddings (adjust based on how you want to merge them)
        #     combined_embeddings = self.embedding_combiner(init_h, llama_embeddings_projected)
        #     # combined_embeddings = init_h + llama_embeddings_projected
        #     # combined_embeddings = init_h + llama_embeddings_projected[:, :init_h.size(1)]
    
        #     h = combined_embeddings
        # else:
        #     h = init_h
        
        
        # for layer in self.layers:
        #     h = layer(h, mask)

        # # https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py#L493
        # if self.post_layers_norm is not None:
        #     h = self.post_layers_norm(h)

        # # Return latent representation
        # return h, init_h  # [B, N, H]
