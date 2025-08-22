#ps.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
from pydantic import BaseModel
from pymongo import MongoClient
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import os
import uuid
import json
import random
from pymongo.errors import PyMongoError

app = FastAPI()
load_dotenv()

MONGO_URI_DEV = os.getenv("MONGO_URI_DEV")
MONGO_URI_PROD = os.getenv("MONGO_URI_PROD")
DB_NAME = os.getenv("DB_NAME", "bacpac")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chatbots")

stage = os.getenv("STAGE", "dev").lower()
mongo_uri = MONGO_URI_PROD if stage == "prod" else MONGO_URI_DEV
if not mongo_uri:
    print("MONGO URI not found in env. Set MONGO_URI_DEV / MONGO_URI_PROD")
    raise RuntimeError("MONGO URI not configured")

print(f"Using Mongo URI for stage='{stage}'")

client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
collection = client[DB_NAME][COLLECTION_NAME]

try:
    client.admin.command("ping")
    print("Connected to MongoDB")
except Exception as e:
    print(f"MongoDB connection failed at startup: {e}")

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set!")
genai.configure(api_key=GEMINI_KEY)

generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

class MessagePayload(BaseModel):
    userId: str
    prompt: str
    threadId: Optional[str] = None
    communityId: Optional[str] = ""
    collegeID: Optional[str] = ""

def generate_full_reply(prompt_text: str) -> str:
    """
    Calls Gemini synchronously (blocking) to get full response text.
    We run this in a thread (asyncio.to_thread) from async code so it doesn't block the event loop.
    """
    try:
        convo = model.start_chat(history=[])
        resp = convo.send_message(prompt_text)  
        text = resp.text or ""
        return text
    except Exception as e:
        print(f"[GEN ERROR] Generation failed: {e}")
        return "Sorry, I couldn't generate a response right now."

# ---- WebSocket Endpoint ----
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")

    try:
        while True:
            raw_data = await websocket.receive_text()
            print(f"[WS] Received raw payload: {raw_data[:200]}")  
            try:
                data = json.loads(raw_data)
            except Exception as e:
                print(f"[WS] Failed to parse JSON payload: {e}")
                await websocket.send_text("Invalid payload format. Expecting JSON.")
                continue

            try:
                payload = MessagePayload(**data)
            except Exception as e:
                print(f"[WS] Payload validation error: {e}")
                await websocket.send_text("Invalid payload fields. Required: userId, prompt")
                continue

            # threadId validation
            if not payload.threadId:
                payload.threadId = f"thread_{uuid.uuid4().hex[:10]}"
                print(f"[THREAD] New thread created: {payload.threadId}")

            # memory blocks + system prompt 
            memory_blocks = []
            try:
                past_msgs = list(collection.find({"threadId": payload.threadId}).sort("createdAt", 1))
                print(f"[DB] Found {len(past_msgs)} past messages for thread {payload.threadId}")
            except Exception as e:
                print(f"[DB ERROR] Failed to read past messages: {e}")
                past_msgs = []

            for msg in past_msgs:
                if msg.get("prompt"):
                    memory_blocks.append(f"User: {msg.get('prompt','')}")
                if msg.get("response"):
                    memory_blocks.append(f"Bot: {msg.get('response','')}")

            memory_blocks.append(f"User: {payload.prompt}")

            system_prompt = """
            You are my academic counsellor with a focus on my college studies.
            Your goal is to provide helpful and encouraging advice on topics like
            time management, study strategies, course selection, and dealing with
            academic stress. Keep your tone supportive and knowledgeable.
            """
            full_prompt = "\n".join([system_prompt.strip(), *memory_blocks])
            print("[GEN] Sending prompt to Gemini (full prompt length:", len(full_prompt), "chars )")

            full_reply = await asyncio.to_thread(generate_full_reply, full_prompt)
            print("[GEN] Received full reply length:", len(full_reply))

            words = full_reply.split()
            i = 0
            accumulated = []
            try:
                while i < len(words):
                    chunk_size = random.choice([6, 7, 8])
                    chunk = " ".join(words[i : i + chunk_size])
                    await websocket.send_text(chunk)
                    accumulated.append(chunk)
                    i += chunk_size
                    # edit needed
                    await asyncio.sleep(random.uniform(0.05, 0.18))
                print("[WS] Finished streaming all chunks to client")
            except WebSocketDisconnect:
                print("[WS] Client disconnected during stream")
                # save what was streamed so far
            except Exception as e:
                print(f"[WS SEND ERROR] {e}")

            final_text = " ".join(accumulated).strip() if accumulated else full_reply
            doc = {
                "userId": payload.userId,
                "communityId": payload.communityId,
                "collegeID": payload.collegeID,
                "prompt": payload.prompt,
                "threadId": payload.threadId,
                "response": final_text,
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow(),
                "__v": 0
            }
            try:
                res = collection.insert_one(doc)
                print(f"[DB] Inserted chat doc with _id={res.inserted_id}")
            except PyMongoError as e:
                print(f"[DB ERROR] Insert failed: {e}")
            except Exception as e:
                print(f"[DB ERROR] Unexpected error on insert: {e}")

    except WebSocketDisconnect:
        print("[WS] Client disconnected (outer loop)")
    except Exception as e:
        print(f"[ERROR] WebSocket handler fatal error: {e}")
        try:
            await websocket.close(code=1011, reason="Internal Server Error")
        except Exception:
            pass

@app.get("/")
async def root():
    return {"message": "Chatbot is running!!"}

@app.post("/stream")
async def stream_response(request: Request):
    try:
        body = await request.json()
        prompt = body.get("prompt")
        user_id = body.get("userId")

        if not prompt or not user_id:
            return JSONResponse({"error": "Missing prompt or userId in request."}, status_code=400)

        full_reply = await asyncio.to_thread(generate_full_reply, prompt)

        async def event_generator():
            words = full_reply.split()
            chunk_size = 6
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.6)

        doc = {
            "userId": user_id,
            "prompt": prompt,
            "response": full_reply,
            "threadId": f"thread_{uuid.uuid4().hex[:10]}",
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow(),
            "__v": 0
        }
        try:
            res = collection.insert_one(doc)
            print(f"[DB] /stream inserted doc _id={res.inserted_id}")
        except Exception as e:
            print(f"[DB ERROR] /stream insert failed: {e}")

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        print(f"[ERROR] /stream handler: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)