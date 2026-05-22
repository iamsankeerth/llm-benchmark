import json
import re
from pydantic import ValidationError
from typing import Optional
from src.ollama_client import OllamaClient
from src.schemas import SCHEMAS, get_schema_for_category


def _extract_json(text: str) -> Optional[str]:
    """Extract the first valid JSON object from text (handles markdown wrapping)."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
        if match:
            text = match.group(1).strip()
    brace_start = text.find("{")
    if brace_start == -1:
        return None
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start:i + 1]
    return None


class StructuredOutputTester:
    def __init__(self):
        self.client = OllamaClient()

    def generate_single_with_retry(
        self,
        model: str,
        prompt: str,
        category: str = "Chat & Generation",
        temperature: float = 0.0,
        max_retries: int = 3,
        image_path: Optional[str] = None
    ):
        """
        Phase 2: Enforce JSON output schema, validate with pydantic, and implement retry mechanism.
        Returns a dictionary of success status and any parsing errors.

        Strategy:
        1. Try Ollama's format=schema (up to max_retries with error feedback)
        2. If format=schema fails completely, fallback to unstructured generation
           and post-process to extract JSON from the response
        """
        schema_class = get_schema_for_category(category)
        schema_json = schema_class.model_json_schema()
        is_vis = (category == "Multimodal Vision")

        # Phase A: Try with format=schema
        last_content = ""
        schema_attempts = 0
        for attempt in range(max_retries + 1):
            schema_attempts += 1
            content, error = self.client.generate_structured(
                model, prompt, schema_json,
                temperature=temperature,
                is_vision=is_vis,
                image_path=image_path
            )
            last_content = content

            if error:
                if attempt == max_retries:
                    break
                continue

            try:
                valid_data = schema_class.model_validate_json(content)
                return {"success": True, "error": None, "output": content}
            except ValidationError as ve:
                prompt += f"\n\nYour previous response failed JSON validation. Ensure it exactly matches the schema. Error: {ve}"

        # Phase B: Fallback — generate without format constraint, extract JSON
        fallback_prompt = (
            f"Return your answer as a JSON object matching this schema:\n"
            f"{json.dumps(schema_json, indent=2)}\n\n"
            f"Your response must be ONLY the JSON object. "
            f"Do not include any text before or after the JSON. "
            f"Do not use markdown code blocks.\n\n"
            f"Original prompt: {prompt.split('Original prompt: ')[-1] if 'Original prompt: ' in prompt else prompt}"
        )

        content, error = self.client.generate_unstructured(
            model, fallback_prompt,
            temperature=temperature,
            is_vision=is_vis,
            image_path=image_path
        )

        if error:
            return {"success": False, "error": f"Schema mode failed ({schema_attempts} attempts), then fallback: {error}", "output": last_content}

        extracted = _extract_json(content)
        if extracted:
            try:
                schema_class.model_validate_json(extracted)
                return {"success": True, "error": None, "output": extracted}
            except ValidationError as ve:
                return {"success": False, "error": f"Fallback JSON invalid: {ve}", "output": extracted}

        return {"success": False, "error": "No JSON found in fallback response", "output": content}
