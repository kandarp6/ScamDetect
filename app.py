import sys
import asyncio
import warnings

if sys.platform == 'win32':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import uvicorn
from loguru import logger

if __name__ == "__main__":
    logger.info("Starting Graphura Unified App (FastAPI + Static Frontend)...")
    logger.info("API Docs: http://localhost:8000/docs")
    logger.info("Frontend: http://localhost:8000/")
    
    try:
        uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
    except KeyboardInterrupt:
        logger.info("Stopping server...")
    except Exception as e:
        logger.error(f"Uvicorn startup failed: {e}")
        sys.exit(1)
