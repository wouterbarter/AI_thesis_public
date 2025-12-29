from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, BitsAndBytesConfig
import torch
import torch.nn.functional as F
import pandas as pd
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache
from src.prompt_manager import PreparedPrompt
from transformers.generation.utils import GenerateOutput
from typing import Union
from typing import Optional
from dataclasses import dataclass



@dataclass
class LogitsContainer:
    """
    Holds model logits.
    - If `indices` is None: Represents the FULL vocabulary (Dense). 
      Values are shape [Seq, Vocab].
    - If `indices` is set: Represents Top-K (Sparse). 
      Values and Indices are shape [Seq, K].
    """
    values: torch.Tensor
    indices: Optional[torch.Tensor] = None
    constrained_values: Optional[torch.Tensor] = None
    constrained_indices: Optional[torch.Tensor] = None

    @property
    def is_sparse(self) -> bool:
        return self.indices is not None
    


@dataclass
class ModelOutput:
    prompt: PreparedPrompt  # The Passthrough: The original input object
    sequence: torch.Tensor
    input_length: int
    logits: Optional[LogitsContainer] = None
    # Optional: error state if something crashed for this specific item
    error: Optional[str] = None


class LogitsMask(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores) -> torch.FloatTensor:
        if not self.allowed_token_ids:
            raise ValueError("No allowed tokens specified for LogitsMask.")
        mask = torch.full_like(scores, float("-inf"), device=scores.device)
        mask[:, list(self.allowed_token_ids)] = 0
        return scores + mask


class Modeler:
    """
    Wrapper around a Hugging Face causal LM for constrained generation and logit inspection.
    """

    def __init__(self, 
                 model_name: str | None = None, 
                 model=None, 
                 tokenizer=None, 
                 device=None, 
                 quantize=True,
                 constrained_generation=False):
        """
        Initialize the modeler with a given model name or existing model/tokenizer.
        """
        self._has_warned_constraints = False # Initialize flag
        self.constrained_generation = constrained_generation # Applies token_constraints from PromptSuite[PromptTemplate]
        # TODO: I am batch streaming which makes it difficult to apply true constrained generation.
        ## For now it is much easier to just slice the 'constrained tokens' from the ModelOutput


        # Choose device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using Apple MPS for GPU acceleration.")
            else:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                print(f"Using device: {self.device}")
        else:
            self.device = device

        # self.device = torch.device('cpu')
        # self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is not None and tokenizer is not None:
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
        elif model_name is not None:
            if quantize:
                # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float32,  # Use float16 for 2070 Super
                    bnb_4bit_use_double_quant=True,     # Saves a bit more memory
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    dtype = torch.float32,
                    device_map="auto"  # This is essential for bitsandbytes
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float32
                ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side='left')
            print(f"Padding side: {self.tokenizer.padding_side}")
        else:
            raise ValueError(
                "Provide either (model_name) or (model, tokenizer) pair.")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        SAFE_DEFAULT_MAX_LENGTH = 8192
        self.model_max_len = self.tokenizer.model_max_length

        if self.model_max_len is None or self.model_max_len > 1e6:
            self.model_max_len = SAFE_DEFAULT_MAX_LENGTH

        self.model.eval()
        # self.tokenizer.padding_side = "left"
        self.allowed_tokens = []
        self.allowed_token_ids = []
        self.logits_processor = None
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def set_token_constraints(self, allowed_tokens: list[str]):
        """
        Restrict generation to the first sub-token of each allowed word.
        This is ideal for deterministic evaluation where the first token is
        enough to identify the intended choice.
        """
        self.allowed_tokens = allowed_tokens
        tokenized = self.tokenizer.convert_tokens_to_ids(allowed_tokens)
        self.allowed_token_ids = sorted(list(set(tokenized)))
        self.allowed_ids_tensor = torch.tensor(self.allowed_token_ids, device=self.device)

        self.logits_processor = LogitsMask(self.allowed_token_ids)


    def clear_token_constraints(self):
        """
        Clears any set token constraints, enabling free generation.
        """
        self.logits_processor = None
        self.allowed_token_ids = []


    def generate_chat(self, prompts: list[PreparedPrompt], max_new_tokens: int = 1, top_k: Optional[int] = None, **gen_kwargs)-> list[ModelOutput]:
        """
        Generates a response using the model's chat template for instruction-following.

        Args: 
            prompts: A batch of PreparedPrompt objects, where each conversation is a list of
                           message dictionaries (e.g., [{"role": "user", "content": "..."}]).
        """
        if not self._has_warned_constraints and self.logits_processor is None:
            print("Warning, token constraints not set!")
            self._has_warned_constraints = True
        
        if top_k is None:
            print("Warning, top_k not set, saving full logits tensor.")

        # 1. Apply the chat template to each conversation
        # This formats the conversation history with all the necessary special tokens.
        # It's crucial to set add_generation_prompt=True!
        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                prompt.conversation,
                tokenize=False,
                add_generation_prompt=True
            ) + prompt.assistant_prefix
            for prompt in prompts
        ]

        # 2. Tokenize the now-formatted prompts
        # The tokenizer will now correctly process the special tokens added by the template.
        model_inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            # padding='max_length',
            truncation=True,
            max_length=self.model_max_len
        ).to(self.device)

        generate_kwargs = {
            **model_inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "output_logits": True,
            "return_dict_in_generate": True, # Should always be True
            **gen_kwargs
        }

        if self.logits_processor is not None:
            # Note: The Hugging Face API requires this to be a list
            generate_kwargs["logits_processor"] = [self.logits_processor]
        

        hf_output = self.model.generate(**generate_kwargs)
        
        input_length = model_inputs['input_ids'].shape[-1]

        return self._process_model_outputs(hf_output=hf_output,
                                    prompts = prompts,
                                    input_length = input_length,
                                    top_k = top_k)
    
    

    def _process_model_outputs(
        self, 
        hf_output, 
        prompts: list[PreparedPrompt], 
        input_length: int, 
        top_k: Optional[int]
            ) -> list[ModelOutput]:
        """
        Internal helper to unbind tensors, apply constraints/compression, 
        and package results into ModelOutput objects.
        """
        
        # 1. Unbind Sequences (Batch -> List)
        batch_sequences = torch.unbind(hf_output.sequences, dim=0)

        # 2. Prepare Logits containers
        batch_logits_containers = []
        
        if hasattr(hf_output, 'logits') and hf_output.logits:
            # Stack to [Batch, Time, Vocab]
            stacked_logits = torch.stack(hf_output.logits, dim=1)
            
            # --- A. Top-K Logic (Batch Operation) ---
            if top_k:
                # Optimized: Run topk on the whole batch at once
                tk_vals, tk_idxs = torch.topk(stacked_logits, k=top_k, dim=-1)
                list_tk_vals = torch.unbind(tk_vals, dim=0)
                list_tk_idxs = torch.unbind(tk_idxs, dim=0)
            else:
                # Dense Mode
                list_tk_vals = torch.unbind(stacked_logits, dim=0)
                list_tk_idxs = [None] * len(prompts)

            # --- B. Constraint Logic (Per-Row Operation) ---
            # We iterate to handle the "Ragged" nature of constraints
            for i, prompt in enumerate(prompts):
                
                # 1. Extract Constraints (if any)
                c_vals = None
                c_idxs = None
                
                if prompt.constraint_ids is not None:
                    # Move target IDs to device
                    targets = prompt.constraint_ids.to(self.device)
                    # Slice the specific row [Seq, V] -> [Seq, C]
                    c_vals = stacked_logits[i].index_select(dim=-1, index=targets)
                    c_idxs = targets # Store the IDs that generated these values

                # 2. Create the Container for this row
                # We have the Top-K parts from the list, and Constrained parts from this loop
                container = LogitsContainer(
                    values=list_tk_vals[i],
                    indices=list_tk_idxs[i],
                    constrained_values=c_vals,
                    constrained_indices=c_idxs
                )
                batch_logits_containers.append(container)
        
        else:
            # Fallback if no logits returned
            batch_logits_containers = [None] * len(prompts)

        # 3. Final Packaging
        results = []
        for prompt, seq, log_container in zip(prompts, batch_sequences, batch_logits_containers):
            results.append(ModelOutput(
                prompt=prompt,
                sequence=seq,
                logits=log_container,
                input_length=input_length
            ))
            
        return results

    def decode(self, model_outputs: list[ModelOutput], 
               skip_special_tokens: bool = True) -> list[str]:
        """
        Decode generated sequences into strings.
        """
        # if not hasattr(self, "model_output"):
        #     raise ValueError("No model output found. Run generate() first.")

        generated_token_list = [
                out.sequence[out.input_length :] 
                for out in model_outputs
            ]

        return self.tokenizer.batch_decode(
                    generated_token_list, 
                    skip_special_tokens=skip_special_tokens
                )


    def get_rating(self, model_output) -> str:
        """
        Return the final rating token (e.g., last generated number/word).
        """
        decoded = self.decode(model_output)[0].strip()
        # extract last whitespace-separated token
        last_token = decoded.split()[-1]
        return last_token

    # def get_logits(self) -> torch.Tensor:
    #     """
    #     Return the raw logits tensor from the last generation.
    #     """
    #     if not hasattr(self, "model_output"):
    #         raise ValueError("No model output found. Run generate() first.")
    #     return self.model_output.logits

    # def get_logits(self) -> torch.Tensor:
    #     """
    #     Return the raw logits tensor from the last generation.
    #     """
    #     if not hasattr(self, "model_output"):
    #         raise ValueError("No model output found. Run generate() first.")
    #     return self.model_output.logits
    
    def get_logits(self, model_output) -> torch.Tensor:
        """
        Return the raw logits tensor from the last generation.
        """
        return model_output.logits

    def get_relevant_logits(self, model_output, normalize: bool = False):
        """
        Return the logits for only the allowed tokens at the final generation step.
        """
        logits = self.get_logits(model_output)[0]  # TODO: incorrect for multiple token generation!
        # shape: (batch, seq_len, vocab)
        relevant_logits = logits[:, self.allowed_token_ids].detach().cpu()

        if normalize:
            relevant_logits = torch.softmax(relevant_logits, dim=0)

        # return {token: float(value) for token, value in zip(self.allowed_tokens, last_logits)}
        return relevant_logits
    
    def get_relevant_logits_dict(self, model_output, normalize: bool = False):
        """
        Return the logits for only the allowed tokens at the FIRST generation step.

        Args:
            model_output: The output from model.generate().
            normalize (bool): If True, apply softmax to convert logits to probabilities.

        Returns:
            List[Dict[str, float]]: A list of dictionaries, one per sample in the batch, 
                                    mapping allowed token names (A, B) to their logit/probability values.
        """
        # shape: (batch, vocab_size)
        first_logits = self.get_logits(model_output)[0]
        # shape: (batch, num_allowed_tokens)
        relevant_logits_batch = first_logits[:, self.allowed_token_ids].detach().cpu()

        relevant_tokens = [self.tokenizer.convert_ids_to_tokens(id) for id in self.allowed_token_ids]

        if normalize:
            relevant_logits_batch = torch.softmax(relevant_logits_batch, dim=1)

        results = []
        for logit_row in relevant_logits_batch:
            # Zip the allowed tokens with the logit values for that single sample
            logit_dict = {
                token: float(value) 
                for token, value in zip(relevant_tokens, logit_row)
            }
            results.append(logit_dict)

        return results

    # Attempts at optimization

    # Should be faster but isnt TODO: check later
    # def run_inference(self, prompts, **kwargs):
    #     inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
    #     with torch.inference_mode():
    #         return self.model(**inputs, use_cache=True, **kwargs)

    # def compute_logits(self, prompts: list[str], **kwargs) -> torch.Tensor:
    #     """
    #     Compute and return the logits for a list of prompts.
    #     Calls run_inference() internally.
    #     """
    #     if self.logits_processor is None:
    #         raise ValueError("You must call set_token_constraints() before compute_logits().")

    #     outputs = self.run_inference(prompts, **kwargs)
    #     return outputs.logits

    # def get_last_logits(self, prompts: list[str]) -> torch.Tensor:
    #     """
    #     Efficiently computes the logits for the next token over a constrained vocabulary.
    #     This is the most efficient method for evaluation tasks.
    #     """

    #     if self.allowed_token_ids is None:
    #         raise ValueError("You must call set_token_constraints() first.")

    #     # 1. Tokenize the entire batch of prompts at once
    #     inputs = self.tokenizer(
    #         prompts,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #     ).to(self.device)

    #     # 2. Perform a single forward pass in inference mode
    #     with torch.inference_mode():
    #         outputs = self.model(**inputs)

    #     # 3. Get the logits for the *last* token of each sequence in the batch
    #     # Logits shape: (batch_size, sequence_length, vocab_size)
    #     last_token_logits = outputs.logits[:, -1, :]
    #     # Sliced shape: (batch_size, vocab_size)

    #     # 4. Filter these logits to include only your allowed tokens
    #     # This returns the final tensor of interest.
    #     relevant_logits = last_token_logits[:, self.allowed_token_ids]
    #     # Final shape: (batch_size, num_allowed_tokens)

    #     return relevant_logits.cpu()

    # def prepare_prompt_cache(self, static_prompt_prefix: str):
    #     """
    #     Processes the static part of the prompt once and stores its KV cache
    #     and its length.
    #     """
    #     inputs = self.tokenizer(static_prompt_prefix, return_tensors="pt").to(self.device)

    #     # Store the length of the prefix ---
    #     self.prefix_token_length = inputs.input_ids.shape[1]

    #     with torch.inference_mode():
    #         outputs = self.model(**inputs)

    #     self.prompt_cache = outputs.past_key_values
    #     print(f"✅ Prompt KV cache for {self.prefix_token_length} tokens has been computed.")

    # def get_evaluation_logits_with_cache(self, dynamic_prompts: list[str]) -> torch.Tensor:
    #     """
    #     Evaluates dynamic prompt parts using the pre-computed KV cache.
    #     """
    #     if not hasattr(self, "prompt_cache"):
    #         raise ValueError("You must call prepare_prompt_cache() first.")

    #     inputs = self.tokenizer(
    #         dynamic_prompts, return_tensors="pt", padding=True, truncation=True
    #     ).to(self.device)

    #     dynamic_attention_mask = inputs.attention_mask

    #     batch_size = inputs.input_ids.shape[0]

    #     # If the batch size is 1, no need to expand the cache
    #     if batch_size == 1:
    #         expanded_cache = self.prompt_cache
    #     else:
    #         # 1. DECONSTRUCT the cache into the simple legacy tuple format
    #         legacy_cache_tuple = self.prompt_cache.to_legacy_cache()
    #         # This gives you a structure like: ((key1, value1), (key2, value2), ...)

    #         # 2. EXPAND the tensors within the simple tuple using a list comprehension
    #         expanded_legacy_cache = tuple(
    #             (
    #                 # Repeat the key tensor (index 0) for the current layer
    #                 layer_cache[0].repeat(batch_size, 1, 1, 1),
    #                 # Repeat the value tensor (index 1) for the current layer
    #                 layer_cache[1].repeat(batch_size, 1, 1, 1)
    #             )
    #             for layer_cache in legacy_cache_tuple
    #         )

    #         # 3. RECONSTRUCT a new, correctly typed DynamicCache object
    #         expanded_cache = DynamicCache.from_legacy_cache(expanded_legacy_cache)

    #     # --- Create and combine the attention mask ---
    #     # Get the length of the sequence already in the cache
    #     static_prefix_len = self.prompt_cache.get_seq_length()

    #     # Create the mask for the static (cached) part
    #     static_attention_mask = torch.ones(batch_size, static_prefix_len, dtype=torch.long).to(self.device)

    #     # Combine the static and dynamic masks to create the full mask
    #     full_attention_mask = torch.cat([static_attention_mask, dynamic_attention_mask], dim=1)

    #     with torch.inference_mode():
    #         outputs = self.model(
    #             input_ids=inputs.input_ids,
    #             past_key_values=expanded_cache,
    #             attention_mask=full_attention_mask  # <-- Pass the full, correct mask
    #         )

    #     # with torch.inference_mode():
    #     #     outputs = self.model(
    #     #         input_ids=inputs.input_ids,
    #     #         past_key_values=expanded_cache  # Now passing a tuple
    #     #     )

    #     last_token_logits = outputs.logits[:, -1, :]
    #     relevant_logits = last_token_logits[:, self.allowed_token_ids]

    #     return relevant_logits.cpu()
