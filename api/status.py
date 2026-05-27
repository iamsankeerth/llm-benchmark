from fastapi import APIRouter, HTTPException

from src.live_status import LiveStatusProjection

router = APIRouter()
projection = LiveStatusProjection()


@router.get("/status")
def get_status():
    return projection.status_payload()


@router.get("/model/{queue_id:path}/prompts")
def get_model_prompts(queue_id: str):
    payload = projection.model_prompts_payload(queue_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"No CSV found for model: {queue_id}")
    return payload


@router.post("/stop")
def stop_pipeline():
    try:
        return projection.stop_pipeline()
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to stop process: {exc}")


@router.get("/health")
def health_check():
    return {"status": "ok"}
