# FastAPI app for FairDecisionLab.

import os
from fastapi import FastAPI
from fastapi.routing import APIRoute

from openenv.core.env_server import create_web_interface_app, create_app

from models import FairAction, FairObservation
from server.environment import FairDecisionEnvironment

if os.environ.get("ENABLE_WEB_INTERFACE", "false").lower() == "true":
    app = create_web_interface_app(FairDecisionEnvironment, FairAction, FairObservation)
else:
    app = create_app(FairDecisionEnvironment, FairAction, FairObservation)

# Remove openenv's /health so ours takes priority
app.router.routes = [r for r in app.router.routes if not (isinstance(r, APIRoute) and r.path == "/health")]


@app.get("/health")
def health():
    return {"status": "ok"}
