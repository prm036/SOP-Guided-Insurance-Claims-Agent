"""
Flask entry point — minimal server that exposes the SOP harness via REST.

Routes:
  POST /api/session       → create a new conversation session
  POST /api/chat          → send a message, receive agent response
  GET  /api/session/<id>  → get session state
  GET  /                  → basic HTML chat page
"""

import os
import logging

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from data.store import DataStore
from llm.client import LLMClient
from agent.harness import SOPHarness

# ── Bootstrap ────────────────────────────────────────────────────────

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-5s  %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Initialise core components ───────────────────────────────────────

data_store = DataStore()

api_key = os.environ.get("OPENAI_API_KEY", "")
model = os.environ.get("LLM_MODEL", "gpt-4o")
base_url = os.environ.get("OPENAI_BASE_URL")

llm_client = LLMClient(api_key=api_key, model=model, base_url=base_url)
harness = SOPHarness(data_store=data_store, llm_client=llm_client)

logger.info("SOP Harness initialised  model=%s", model)


# ── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the basic HTML chat page."""
    return render_template("index.html")


@app.route("/api/session", methods=["POST"])
def create_session():
    """Create a new conversation session."""
    state = harness.create_session()
    logger.info("Session created: %s", state.session_id)
    return jsonify(state.to_dict())


@app.route("/api/session/<session_id>", methods=["GET"])
def get_session(session_id):
    """Return the current session state."""
    state = harness.get_session(session_id)
    if not state:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(state.to_dict())


@app.route("/api/chat", methods=["POST"])
def chat():
    """Process a user message through the SOP harness.

    Expects JSON: { "session_id": "...", "message": "..." }
    Returns JSON:  { "response": "...", "phase": "...", "state": {...} }
    """
    body = request.get_json(force=True)
    session_id = body.get("session_id")
    message = body.get("message", "").strip()

    if not session_id or not message:
        return jsonify({"error": "session_id and message are required"}), 400

    logger.info("Chat  session=%s  phase=%s  msg=%s",
                session_id,
                (harness.get_session(session_id).current_phase.value
                 if harness.get_session(session_id) else "?"),
                message[:80])

    try:
        result = harness.process_message(session_id, message)
    except Exception:
        logger.exception("Error processing message")
        return jsonify({"error": "Internal error processing your message"}), 500

    return jsonify(result)


@app.route("/api/config", methods=["GET", "POST"])
def config_endpoint():
    """Check or set API key at runtime."""
    if request.method == "GET":
        has_key = bool(llm_client.api_key)
        return jsonify({"has_key": has_key})
        
    body = request.get_json(force=True)
    key = body.get("api_key", "").strip()
    if key:
        llm_client.set_api_key(key)
        logger.info("API key updated via /api/config")
        return jsonify({"status": "ok"})
    return jsonify({"error": "api_key is required"}), 400


# ── Run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
