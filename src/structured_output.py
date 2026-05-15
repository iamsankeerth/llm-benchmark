import json
from pydantic import ValidationError
from typing import Optional
from src.ollama_client import OllamaClient
from src.schemas import SCHEMAS, get_schema_for_category

class StructuredOutputTester:
    def __init__(self):
        self.client = OllamaClient()

    def generate_single_with_retry(
        self,
        model: str,
        prompt: str,
        category: str = "Chat & Generation",
        temperature: float = 0.0,
        max_retries: int = 1,
        image_path: Optional[str] = None
    ):
        """
        Phase 2: Enforce JSON output schema, validate with pydantic, and implement retry mechanism.
        Returns a dictionary of success status and any parsing errors.
        """
        schema_class = get_schema_for_category(category)
        schema_json = schema_class.model_json_schema()
        is_vis = (category == "Multimodal Vision")

        for attempt in range(max_retries + 1):
            content, error = self.client.generate_structured(
                model, prompt, schema_json,
                temperature=temperature,
                is_vision=is_vis,
                image_path=image_path
            )

            if error:
                if attempt == max_retries:
                    return {"success": False, "error": error, "output": content}
                continue

            try:
                valid_data = schema_class.model_validate_json(content)
                return {"success": True, "error": None, "output": content}
            except ValidationError as ve:
                prompt += f"\n\nYour previous response failed JSON validation. Ensure it exactly matches the schema. Error: {ve}"

        return {"success": False, "error": "Max retries exceeded", "output": content}
