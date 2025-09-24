from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .model import MCPClientModelRequest, MCPClientModelResponse

from .client import handle_input, download_ollama_models, is_model_downloaded
from .client import ModelStatus
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler: runs once on startup and shutdown.

    We run the blocking model download in a thread to avoid blocking the
    event loop, matching previous behavior from the @app.on_event startup
    handler.
    """
    print("MCP Client API is starting up...")
    # Start the blocking downloader in a background task (fire-and-forget)
    # so the app becomes responsive immediately.
    asyncio.create_task(asyncio.to_thread(download_ollama_models))
    print("Ollama models download started in background (async task).")
    yield
    # Optional shutdown actions
    print("MCP Client API is shutting down...")

app = FastAPI(lifespan=lifespan)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the MCP Client API!"}

@app.post("/process")
async def process_data(data : MCPClientModelRequest) -> MCPClientModelResponse:
    """
    Process the input data and return a response.
    """
    response_message = await handle_input(data.user_query)
    return MCPClientModelResponse(response=response_message)

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    Handles ModelStatus enum from is_model_downloaded().
    """

    try:
        # is_model_downloaded is synchronous; run it in a thread to avoid
        # blocking the event loop
        status = await asyncio.to_thread(is_model_downloaded)
        if status == ModelStatus.DOWNLOADED:
            return {"status": "healthy", "message": "Services are up and ready to serve."}
        elif status == ModelStatus.NOT_FOUND:
            return {"status": "wait", "message": "Service is initializing, please wait."}
        elif status == ModelStatus.ERROR:
            return {"status": "error", "message": "Service is unavailable, please try again later."}
        else:
            return {"status": "unknown", "message": f"Unexpected status: {status}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# (Startup handled by lifespan handler above)