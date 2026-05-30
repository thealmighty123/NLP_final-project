import os
import re
import time
from dataclasses import dataclass
from typing import List, Any, Optional, Literal
import json

from openai import OpenAI
from source.module.generate.base import BaseGenerator, BaseGeneratorConfig


@dataclass
class OpenRouterGeneratorConfig(BaseGeneratorConfig):
    model_name: Optional[str] = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
    max_new_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.0

    # thêm để tránh lỗi khi train.py truyền từ config Llama cũ
    max_total_tokens: Optional[int] = None
    min_new_tokens: Optional[int] = None
    use_vllm: Optional[bool] = None
    gpu: Optional[int] = None
    tensor_parallel_size: Optional[int] = None

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

    def _is_retryable_error(self, error: Exception) -> bool:
        error_text = str(error).lower()
        return (
            "429" in error_text
            or "rate" in error_text
            or "503" in error_text
            or "capacity_error" in error_text
            or "no backends available" in error_text
            or "provider returned error" in error_text
        )

    def _retry_wait(self, attempt: int) -> int:
        return min(120, 35 + attempt * 20)

    def _generate(self, input_texts: List[str]) -> List[str]:
        outputs = []

        for prompt in input_texts:
            for attempt in range(8):
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
                    if self._is_retryable_error(e):
                        wait = self._retry_wait(attempt)
                        print(f"OpenRouter temporary error. Waiting {wait}s before retry...")
                        time.sleep(wait)
                    elif self._is_moderation_error(e):
                        print("OpenRouter moderation blocked a generation request. Returning Unknown.")
                        completion = None
                        break
                    else:
                        raise
            else:
                raise RuntimeError("OpenRouter temporary error persisted after retries.")

            if completion is None:
                outputs.append('{"answer": "Unknown"}')
            else:
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
    
    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

        return json.loads(text)

    def _is_moderation_error(self, error: Exception) -> bool:
        error_text = str(error).lower()
        return (
            "403" in error_text
            and (
                "moderation" in error_text
                or "flagged" in error_text
                or "sexual/minors" in error_text
            )
        )


    def score_document(self, question: str, answer: str, document: str) -> float:
        prompt = f"""
    You are judging whether a document is useful for multi-hop question answering.

    Question:
    {question}

    Correct answer:
    {answer}

    Document:
    {document}

    Return only valid JSON:
    {{"relevance": 0.0, "consistency": 0.0}}

    Definitions:
    - relevance: whether the document is related to the question.
    - consistency: whether the document supports the correct answer.
    Use numbers from 0 to 1.
    """

        for attempt in range(8):
            try:
                completion = self.client.chat.completions.create(
                    model=self.cfg.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Return only valid JSON. "
                                "No markdown. No explanation. "
                                "The JSON must contain exactly two fields: "
                                "relevance and consistency."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0.0,
                    max_tokens=128,
                )
                break

            except Exception as e:
                if self._is_retryable_error(e):
                    wait = self._retry_wait(attempt)
                    print(f"OpenRouter temporary error. Waiting {wait}s before retry...")
                    time.sleep(wait)
                elif self._is_moderation_error(e):
                    print("OpenRouter moderation blocked a judge request. Using score 0.0.")
                    return 0.0
                else:
                    raise
        else:
            raise RuntimeError("OpenRouter temporary error persisted after retries.")

        text = completion.choices[0].message.content

        if text is None:
            return 0.0

        try:
            data = self._extract_json(text)

            relevance = float(data.get("relevance", 0.0))
            consistency = float(data.get("consistency", 0.0))

            relevance = max(0.0, min(1.0, relevance))
            consistency = max(0.0, min(1.0, consistency))

            return relevance * consistency

        except Exception:
            print("Failed to parse judge output:", text)
            return 0.0
