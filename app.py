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
    import os
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    
    logger.info(f"Starting Graphura Unified App (FastAPI + Static Frontend) on {host}:{port}...")
    logger.info(f"API Docs: http://{host}:{port}/docs")
    logger.info(f"Frontend: http://{host}:{port}/")
    
    try:
        uvicorn.run("backend.main:app", host=host, port=port, reload=False)
    except KeyboardInterrupt:
        logger.info("Stopping server...")
    except Exception as e:
        logger.error(f"Uvicorn startup failed: {e}")
        sys.exit(1)
