from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from api.routes import router
from api.status import router as status_router
import os

app = FastAPI(title="Local AI Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(status_router, prefix="/api")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Local AI Assistant API is running. Check /docs."}

@app.get("/live")
def live_dashboard():
    html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard", "live.html")
    return FileResponse(html_path, media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
