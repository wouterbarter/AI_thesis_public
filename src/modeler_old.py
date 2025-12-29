from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, BitsAndBytesConfig
import torch
import torch.nn.functional as F
import pandas as pd
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache


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

    def __init__(self, model_name: str = None, model=None, tokenizer=None, device=None, quantize=True):
        """
        Initialize the modeler with a given model name or existing model/tokenizer.
        """

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
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    # dtype=torch.float16
                    quantization_config=quantization_config)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16
                ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            raise ValueError(
                "Provide either (model_name) or (model, tokenizer) pair.")

        self.model.eval()
        self.tokenizer.padding_side = "left"
        self.allowed_tokens = None
        self.allowed_token_ids = None
        self.logits_processor = None

    def set_token_constraints(self, allowed_tokens: list[str]):
        """
        Restrict generation to a fixed set of allowed tokens.
        """
        self.allowed_tokens = allowed_tokens
        tokenized = self.tokenizer(
            allowed_tokens, add_special_tokens=False).input_ids

        # Flatten and validate
        flattened = []
        for token, ids in zip(allowed_tokens, tokenized):
            if len(ids) != 1:
                raise ValueError(
                    f"Token '{token}' maps to {len(ids)} sub-tokens: {ids}")
            flattened.extend(ids)

        self.allowed_token_ids = flattened
        self.logits_processor = LogitsMask(self.allowed_token_ids)

    def generate(self, prompts: list[str], max_new_tokens: int = 1, **gen_kwargs):
        """
        Generate output from the model with optional logits masking.
        """
        if self.logits_processor is None:
            raise ValueError(
                "You must call set_token_constraints() before generate().")

        # model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        self.prompts = prompts

        model_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        self.model_output = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            logits_processor=[self.logits_processor],
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
            **gen_kwargs
        )

    def decode(self, skip_special_tokens: bool = True) -> list[str]:
        """
        Decode generated sequences into strings.
        """
        if not hasattr(self, "model_output"):
            raise ValueError("No model output found. Run generate() first.")
        return self.tokenizer.batch_decode(self.model_output.sequences, skip_special_tokens=skip_special_tokens)

    def get_rating(self) -> str:
        """
        Return the final rating token (e.g., last generated number/word).
        """
        decoded = self.decode()[0].strip()
        # extract last whitespace-separated token
        last_token = decoded.split()[-1]
        return last_token

    def get_logits(self) -> torch.Tensor:
        """
        Return the raw logits tensor from the last generation.
        """
        if not hasattr(self, "model_output"):
            raise ValueError("No model output found. Run generate() first.")
        return self.model_output.logits

    def get_relevant_logits(self, normalize: bool = False) -> dict[str, float]:
        """
        Return the logits for only the allowed tokens at the final generation step.
        """
        logits = self.get_logits(
        )[0]  # TODO: incorrect for multiple token generation!
        # shape: (batch, seq_len, vocab)
        relevant_logits = logits[:, self.allowed_token_ids].detach().cpu()

        if normalize:
            relevant_logits = torch.softmax(last_logits, dim=0)

        # return {token: float(value) for token, value in zip(self.allowed_tokens, last_logits)}
        return relevant_logits

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
