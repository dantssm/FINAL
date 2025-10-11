from fastapi import WebSocket, WebSocketDisconnect
import json
from src.pipeline.deep_search import DeepSearchPipeline

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_progress(self, websocket: WebSocket, message: dict):
        await websocket.send_json(message)

manager = ConnectionManager()