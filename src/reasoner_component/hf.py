"""HuggingFace local model generator."""

import os
import sys
import subprocess
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseGenerator

# Transformers version requirements per model.
_MODEL_TRANSFORMERS_REQUIREMENTS: dict[str, str] = {
    "openbmb/AgentCPM-Report": ">=4.46.0,<5.0.0",
    "rl-research/DR-Tulu-8B":  ">=4.52.0,<5.0.0",
}


def ensure_transformers_version(model_name: str) -> None:
    """Ensure the installed transformers version satisfies the model's requirements.

    Installs a compatible version and restarts the process via os.execv if needed.
    Call this *before* constructing an ``HFGenerator`` (the factory does this).
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


class HFGenerator(BaseGenerator):
    """LLM generator for locally loaded HuggingFace models.

    Supports any HF causal-LM model. Known finetuned models
    (openbmb/AgentCPM-Report, rl-research/DR-Tulu-8B) have their
    required transformers version enforced automatically.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        max_input_tokens: int = 4096,
        **kwargs,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_input_tokens = max_input_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        from utils.token_meter import TokenMeter
        self.token_meter = TokenMeter()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()

    def complete(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        return self._generate(messages, temperature, max_tokens)

    async def acomplete(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(
                executor, self._generate, messages, temperature, max_tokens
            )

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        try:
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
                truncation=True,
                max_length=self.max_input_tokens,
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
                if temperature < 0.01:
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                else:
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

            generated = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            self.token_meter.record(input_len, int(outputs.shape[1]) - input_len)

            # Truncate at </action> tag if present (AgentCPM format)
            action_end = generated.find("</action>")
            if action_end != -1:
                generated = generated[:action_end + len("</action>")]

            return generated.strip()
        except Exception:
            import traceback
            traceback.print_exc()
            return ""

    def cleanup(self) -> None:
        if self.device == "cuda":
            torch.cuda.empty_cache()
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
