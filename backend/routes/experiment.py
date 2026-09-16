from fastapi import APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks
from pydantic import BaseModel
import asyncio
import json
import uuid
from typing import Optional

# SecureMesh-SCE Imports (Research-Grade)
from securemesh_sce.experiments.engine import SCENEExperimentEngine, ExperimentConfig
from securemesh_sce.game.actions import AttackerAction, DefenderAction

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                pass

manager = ConnectionManager()

class ExperimentRequest(BaseModel):
    attacker: str
    defender: str
    scenario: str
    episodes: int = 5

def run_experiment_task(request: ExperimentRequest, loop: asyncio.AbstractEventLoop):
    spec_id = f"live-{uuid.uuid4().hex[:8]}"
    config = ExperimentConfig(
        id=spec_id,
        name="Live Web Experiment",
        attacker_name=request.attacker,
        defender_name=request.defender,
        episodes=request.episodes,
        duration=60,
    )
    engine = SCENEExperimentEngine(config=config)

    def on_event(event: dict):
        cb_event = event.copy()
        if 'attacker_action' in cb_event and isinstance(cb_event['attacker_action'], AttackerAction):
            cb_event['attacker_action'] = cb_event['attacker_action'].name
        if 'defender_action' in cb_event and isinstance(cb_event['defender_action'], DefenderAction):
            cb_event['defender_action'] = cb_event['defender_action'].name
            
        asyncio.run_coroutine_threadsafe(
            manager.broadcast({"type": "step", "data": cb_event}),
            loop
        )

    def on_episode(episode: int, metrics: dict):
        asyncio.run_coroutine_threadsafe(
            manager.broadcast({"type": "episode_summary", "episode": episode, "metrics": metrics}),
            loop
        )

    # Broadcast start
    asyncio.run_coroutine_threadsafe(
        manager.broadcast({"type": "status", "message": "Experiment started", "spec": {"id": spec_id, "attacker": request.attacker, "defender": request.defender}}),
        loop
    )
    
    # Run engine synchronously in background thread
    try:
        summary = engine.run(event_callback=on_event, episode_callback=on_episode)
        asyncio.run_coroutine_threadsafe(
            manager.broadcast({"type": "status", "message": "Experiment completed", "metrics": summary.get("aggregate_metrics", {})}),
            loop
        )
    except Exception as e:
        asyncio.run_coroutine_threadsafe(
            manager.broadcast({"type": "status", "message": f"Experiment failed: {str(e)}"}),
            loop
        )


@router.post("/start")
async def start_experiment(request: ExperimentRequest, background_tasks: BackgroundTasks):
    loop = asyncio.get_running_loop()
    background_tasks.add_task(run_experiment_task, request, loop)
    return {"status": "started", "request": request}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
