from fastapi import APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks
from pydantic import BaseModel
import asyncio
import json
import uuid

# SecureMesh-SCE Imports
from securemesh_testbed.experiments.engine import SCEEngine
from securemesh_testbed.experiments.schemas import ExperimentSpec
from securemesh_testbed.agents.attacker import AttackerAction
from securemesh_testbed.agents.defender import DefenderAction

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
        # Broadcast asynchronously
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                pass

manager = ConnectionManager()

class ExperimentRequest(BaseModel):
    attacker: str
    defender: str
    scenario: str
    episodes: int = 5

def run_experiment_task(request: ExperimentRequest, loop: asyncio.AbstractEventLoop):
    engine = SCEEngine()
    spec = ExperimentSpec(
        id=f"live-{uuid.uuid4().hex[:8]}",
        name="Live Web Experiment",
        attacker=request.attacker,
        defender=request.defender,
        scenario=request.scenario,
        episodes=request.episodes,
        duration=100
    )

    def on_event(event: dict):
        # Convert enums to names if they are enums
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
        manager.broadcast({"type": "status", "message": "Experiment started", "spec": spec.model_dump() if hasattr(spec, 'model_dump') else spec.dict()}),
        loop
    )
    
    # Run engine synchronously in background thread
    try:
        record = engine.run(spec, event_callback=on_event, episode_callback=on_episode)
        asyncio.run_coroutine_threadsafe(
            manager.broadcast({"type": "status", "message": "Experiment completed", "metrics": record.metrics}),
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
            data = await websocket.receive_text()
            # Simple ping/pong if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)
