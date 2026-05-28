import os
import re
import time
from dataclasses import dataclass
from typing import List, Any, Optional, Literal

from openai import OpenAI
from source.module.generate.base import BaseGenerator, BaseGeneratorConfig


@dataclass
class OpenRouterGeneratorConfig(BaseGeneratorConfig):
    model_name: Optional[str] = os.getenv("OPENROUTER_MODEL", "openrouter/free")
    max_new_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.0


class OpenRouterGenerator(BaseGenerator):
    def __init__(self, cfg: OpenRouterGeneratorConfig = OpenRouterGeneratorConfig()):
        super().__init__(cfg)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def _clean_output(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\\s*", "", text)
        text = re.sub(r"\\s*```$", "", text)
        return text.strip()

    def _generate(self, input_texts: List[str]) -> List[str]:
        outputs = []

        for prompt in input_texts:
            for attempt in range(5):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.cfg.model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Return exactly one valid JSON object and nothing else. "
                                    "If asked for an answer, return {\"answer\": \"...\"}. "
                                    "If unknown, return {\"answer\": \"Unknown\"}. "
                                    "If asked for a thought, return {\"thought\": \"...\"}. "
                                    "No markdown. No explanation."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self.cfg.temperature,
                        max_tokens=self.cfg.max_new_tokens,
                    )
                    break
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        time.sleep(35 + 10 * attempt)
                    else:
                        raise
            else:
                raise RuntimeError("OpenRouter rate limit persisted after retries.")

            outputs.append(self._clean_output(completion.choices[0].message.content or ""))

        return outputs

    def _score(
        self,
        input_texts: List[str],
        output_texts: List[str],
        method: Literal["perplexity_score"] = "perplexity_score",
    ) -> Any:
        raise NotImplementedError(
            "API generator not supported token-level scoring. "
            "Want to train ReSCORE with the correct formula, need to modify the scoring part."
        )