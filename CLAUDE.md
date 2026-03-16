# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VoicEra is a voice AI platform for managing AI-powered phone agents. It consists of three main services: a FastAPI backend, a Pipecat-based voice server for real-time telephony, and a Next.js frontend dashboard.

## Architecture

```
Frontend (Next.js :3000) → Nginx (:8080) → Backend (FastAPI :8000)
                                         → Voice Server (Pipecat :7860)
                                         → MinIO (:9000)
MongoDB (:27017) ← Backend, Voice Server
```

- **voicera_backend/** — FastAPI REST API. Handles users, agents, campaigns, meetings, audiences, recordings, phone numbers, analytics. JWT auth, MongoDB, MinIO storage.
- **voice_2_voice_server/** — Pipecat pipeline (STT → LLM → TTS) with Vobiz telephony integration. Service factory pattern in `api/services.py` supports multiple providers. System prompts use `{{variable}}` placeholders for dynamic injection.
- **voicera_frontend/** — Next.js 16 (App Router) with Radix UI, TailwindCSS 4, Wavesurfer.js for audio, Recharts for analytics. API routes in `app/api/` proxy to the backend.
- **ai4bharat_stt_server/** and **ai4bharat_tts_server/** — Optional local Indic language STT/TTS services.

### Voice Server Provider Support

STT: Deepgram, AI4Bharat IndicConformer, Bhashini, Sarvam, and more
TTS: Cartesia, AI4Bharat IndicParler, Bhashini, Sarvam, and more
LLM: OpenAI, Google Gemini, Anthropic, KenpathLLM (custom)

Provider mappings are in `voice_2_voice_server/config/{llm,stt,tts}_mappings.py`.

## Common Commands

### Docker (full stack)
```bash
make build-all-services      # Build all Docker services
make start-all-services      # Start everything
make stop-all-services       # Stop everything
make start-backend-services  # Start only MongoDB, MinIO, backend, nginx
make stop-all-ports          # Force kill ports 3000, 27017, 8000, 8001, 8002, 7860
```

### Local development
```bash
# Backend infrastructure (MongoDB + MinIO via Docker)
make start-backend-services

# Frontend
cd voicera_frontend && npm install && npm run dev

# Voice server
cd voice_2_voice_server && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && python main.py

# Backend (without Docker)
cd voicera_backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd voicera_frontend
npm run dev       # Development server
npm run build     # Production build
npm run lint      # ESLint
```

### API docs
- Backend Swagger: `http://localhost:8000/docs`
- Voice Server Swagger: `http://localhost:7860/docs`

## Key Patterns

- **Multi-tenancy**: Most MongoDB collections use `org_id` for tenant isolation.
- **Service factory**: `voice_2_voice_server/api/services.py` creates STT/TTS/LLM service instances by provider name string.
- **Latency optimization**: Voice server monkeypatches SOXR resampler for "Quick Quality" and uses `FastPunctuationAggregator` for text streaming.
- **Telephony flow**: Vobiz webhook → Voice Server `/answer` endpoint → WebSocket audio stream → Pipecat pipeline.
- **Database init**: Collections and indexes are auto-created at backend startup via `voicera_backend/app/database_init.py`.
- **Frontend API proxy**: Next.js API routes (`voicera_frontend/app/api/`) forward requests to the FastAPI backend.

## Environment

Copy `.env.example` to `.env`. Required keys include Vobiz telephony credentials, at least one LLM API key (OpenAI/Gemini/Anthropic), Deepgram for STT, and Cartesia for TTS. Default audio sample rate is 8000 Hz (PSTN standard).
