from pydantic import BaseModel, Field

class Complexity(BaseModel):
    time: str = Field(description="Time complexity of the solution")
    space: str = Field(description="Space complexity of the solution")

class CodingGeneration(BaseModel):
    task_summary: str = Field(description="Summary of the coding task")
    approach: str = Field(description="Approach taken to solve the problem")
    code: str = Field(description="The code implementation")
    complexity: Complexity = Field(description="Time and space complexity")
    edge_cases: list[str] = Field(description="Edge cases considered")
    tests: list[str] = Field(description="Tests for the code")
    risks: list[str] = Field(description="Potential risks or limitations")

class MediumReasoning(BaseModel):
    answer: str = Field(description="Final answer or conclusion")
    reasoning_steps: list[str] = Field(description="Step-by-step reasoning process")
    assumptions: list[str] = Field(description="Assumptions made during reasoning")
    counterarguments: list[str] = Field(description="Counterarguments considered")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")

class ChatGeneration(BaseModel):
    intent: str = Field(description="Intent of the message")
    response: str = Field(description="The generated response")
    tone: str = Field(description="Tone of the response")
    key_points: list[str] = Field(description="Key points in the response")
    constraints_followed: list[str] = Field(description="Constraints that were followed")
    risks_or_caveats: list[str] = Field(description="Risks or caveats in the response")

class MultimodalVision(BaseModel):
    visual_summary: str = Field(description="Summary of the visual content")
    observations: list[str] = Field(description="Observations from the image")
    extracted_text: list[str] = Field(description="Extracted text from the image")
    spatial_relationships: list[str] = Field(description="Spatial relationships observed in the image")
    uncertainties: list[str] = Field(description="Uncertainties in the analysis")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")

SCHEMAS = {
    "CodingGeneration": CodingGeneration,
    "MediumReasoning": MediumReasoning,
    "ChatGeneration": ChatGeneration,
    "MultimodalVision": MultimodalVision,
}

def get_schema_for_category(category: str):
    """Get the appropriate Pydantic schema class for a given benchmark category."""
    mapping = {
        "Coding Generation": CodingGeneration,
        "Medium Reasoning": MediumReasoning,
        "Chat & Generation": ChatGeneration,
        "Multimodal Vision": MultimodalVision,
    }
    return mapping.get(category, ChatGeneration)
