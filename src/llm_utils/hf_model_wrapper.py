"""
Hugging Face model wrapper for local inference.

This module provides a wrapper for the AgentCPM-Report model from Hugging Face
that mimics the LiteLLMClient interface used by AgentCPM.
"""

import os
import sys
import subprocess
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# Transformers version requirements per finetuned model.
# AgentCPM-Report (MiniCPM) uses AttentionMaskConverter from the 4.x API.
# DR-Tulu-8B (Qwen3) was built and tested with transformers 4.52.4.
# Both are incompatible with transformers 5.x.
_MODEL_TRANSFORMERS_REQUIREMENTS: dict[str, str] = {
    "openbmb/AgentCPM-Report": ">=4.46.0,<5.0.0",
    "rl-research/DR-Tulu-8B":  ">=4.52.0,<5.0.0",
}


def _check_and_install_transformers(model_name: str) -> None:
    """Ensure the installed transformers version satisfies the model's requirements.

    If the current version is incompatible, installs the correct version and
    restarts the script (via os.execv) so the new version is picked up cleanly.
    """
    req = _MODEL_TRANSFORMERS_REQUIREMENTS.get(model_name)
    if req is None:
        return

    from importlib.metadata import version as pkg_version
    from packaging.version import Version
    from packaging.specifiers import SpecifierSet

    current = Version(pkg_version("transformers"))
    if current in SpecifierSet(req):
        return

    print(
        f"[HF] transformers {current} does not satisfy '{req}' required by {model_name}.\n"
        f"[HF] Installing a compatible version..."
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", f"transformers{req}", "--quiet"],
        stdout=subprocess.DEVNULL,
    )
    print("[HF] Restarting script with the updated transformers version...")
    os.execv(sys.executable, [sys.executable] + sys.argv)


class HuggingFaceModelWrapper:
    """
    Wrapper for Hugging Face models that provides the same interface as LiteLLMClient.

    This allows using fine-tuned models from Hugging Face (like openbmb/AgentCPM-Report)
    as a drop-in replacement for API-based LLMs.
    """

    def __init__(
        self,
        model_name: str = "openbmb/AgentCPM-Report",
        device: str = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        **kwargs
    ):
        """
        Initialize the Hugging Face model wrapper.

        Args:
            model_name: Hugging Face model identifier (e.g., "openbmb/AgentCPM-Report")
            device: Device to load model on ("cuda", "cpu", or None for auto-detect)
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments (for compatibility with LiteLLMClient)
        """
        _check_and_install_transformers(model_name)

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()

    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format messages using the tokenizer's chat template.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            Formatted prompt string
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    async def acomplete(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Async completion method (compatible with LiteLLMClient interface).

        Args:
            messages: List of message dictionaries
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Generated text string
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        # Use provided values or defaults
        temp = temperature if temperature is not None else self.temperature
        max_len = max_tokens if max_tokens is not None else self.max_tokens

        # Define the synchronous generation function
        def _generate():
            try:
                # Apply chat template and tokenize in one step to ensure special tokens
                # are encoded correctly. The two-step approach (apply_chat_template with
                # tokenize=False then re-tokenizing) can cause special tokens like
                # <|im_start|> to be incorrectly encoded, making the model treat the
                # input as plain text and repeat/continue the prompt instead of answering.
                tokenized = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    truncation=True,
                    max_length=4096,
                )
                # apply_chat_template returns a Tensor or BatchEncoding depending on
                # the transformers version; handle both and preserve attention_mask.
                if hasattr(tokenized, "keys"):
                    input_ids = tokenized["input_ids"].to(self.device)
                    attention_mask = tokenized.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                else:
                    input_ids = tokenized.to(self.device)
                    attention_mask = None

                input_len = input_ids.shape[1]

                with torch.no_grad():
                    if temp < 0.01:  # Very low temperature - use greedy decoding
                        outputs = self.model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_len,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_len,
                            do_sample=True,
                            temperature=temp,
                            top_p=0.95,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )

                generated_text = self.tokenizer.decode(
                    outputs[0][input_len:],
                    skip_special_tokens=True
                )

                # Truncate at </action> tag if present (AgentCPM format)
                action_end = generated_text.find("</action>")
                if action_end != -1:
                    generated_text = generated_text[:action_end + len("</action>")]

                return generated_text.strip()
            except Exception as e:
                import traceback
                traceback.print_exc()
                return ""

        # Run generation in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(executor, _generate)

        return result

    def complete(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Synchronous completion method (compatible with LiteLLMClient interface).

        Args:
            messages: List of message dictionaries
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Generated text string
        """
        import asyncio

        # Run async method in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.acomplete(messages, temperature, max_tokens, **kwargs)
        )

    def cleanup(self):
        """
        Cleanup method (compatible with LiteLLMClient interface).
        """
        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Delete model and tokenizer to free memory
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
