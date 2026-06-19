from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import anthropic
import json
import re

app = FastAPI(title="PROSPECTIVEAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic()

# ── Models ──────────────────────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    transcript: str
    team_members: Optional[List[str]] = []
    meeting_title: Optional[str] = "Team Meeting"
    mood_checkins: Optional[dict] = {}

class TextRequest(BaseModel):
    text: str

# ── Helper ───────────────────────────────────────────────────────────────────

def parse_json_from_response(text: str) -> dict:
    """Extract JSON from Claude's response safely."""
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "PROSPECTIVEAI API running", "version": "1.0.0"}


@app.post("/analyse")
async def analyse_transcript(req: AnalyseRequest):
    """Core analysis — echo score, blind spots, devil's advocate, dominance map."""
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty")

    prompt = f"""
You are PROSPECTIVEAI, an expert at detecting groupthink, echo chambers, and blind spots in team meetings.

Analyse this meeting transcript and return ONLY valid JSON (no markdown, no preamble):

Meeting Title: {req.meeting_title}
Team Members: {', '.join(req.team_members) if req.team_members else 'Unknown'}
Transcript:
\"\"\"
{req.transcript}
\"\"\"

Return this exact JSON structure:
{{
  "echo_score": <integer 0-100, where 0=healthy debate, 100=pure groupthink>,
  "echo_label": "<Low|Moderate|High|Critical>",
  "summary": "<2-3 sentence meeting summary>",
  "unchallenged_ideas": [
    {{"idea": "<idea>", "risk": "<why this is risky unchallenged>"}}
  ],
  "blind_spots": [
    {{"topic": "<topic>", "description": "<what was missed>"}}
  ],
  "devils_advocate": [
    {{"point": "<counterargument AI generates>"}}
  ],
  "dominance_map": [
    {{"name": "<speaker name or Speaker 1>", "percentage": <talk time %>,"role": "<Dominant|Active|Passive|Silent>"}}
  ],
  "question_gaps": [
    {{"question": "<important question nobody asked>"}}
  ],
  "opinion_shifts": "<description of whether anyone changed their mind>",
  "action_items": [
    {{"task": "<task>", "owner": "<person>", "deadline": "<suggested deadline>"}}
  ],
  "positive_highlights": [
    {{"highlight": "<something the team did well>"}}
  ],
  "recommendation": "<one key recommendation to improve team thinking>"
}}
"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    result = parse_json_from_response(message.content[0].text)
    return {"success": True, "data": result}


@app.post("/devil-advocate")
async def devil_advocate(req: TextRequest):
    """Generate devil's advocate counterarguments for a specific idea."""
    prompt = f"""
Generate 3 strong devil's advocate counterarguments for this idea or decision.
Return ONLY JSON, no markdown:

Idea: "{req.text}"

{{
  "counterarguments": [
    {{"point": "<counterargument>", "severity": "<Low|Medium|High>", "suggestion": "<what to consider>"}}
  ]
}}
"""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    result = parse_json_from_response(message.content[0].text)
    return {"success": True, "data": result}


@app.post("/what-if")
async def what_if_generator(req: TextRequest):
    """Generate 'what if' perspective-shifting questions."""
    prompt = f"""
Generate 4 powerful "What If" questions that challenge assumptions about this topic.
Return ONLY JSON, no markdown:

Topic: "{req.text}"

{{
  "what_ifs": [
    {{"question": "<what if question>", "perspective": "<whose lens this is from>"}}
  ]
}}
"""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    result = parse_json_from_response(message.content[0].text)
    return {"success": True, "data": result}


@app.post("/perspective-cards")
async def perspective_cards(req: TextRequest):
    """Generate perspective cards for a meeting topic."""
    prompt = f"""
Generate 4 perspective cards for a team meeting about this topic.
Each card gives a different stakeholder's viewpoint.
Return ONLY JSON, no markdown:

Topic: "{req.text}"

{{
  "cards": [
    {{"role": "<stakeholder role>", "perspective": "<their viewpoint>", "key_concern": "<their main concern>", "color": "<one of: blue|gold|green|red>"}}
  ]
}}
"""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}]
    )
    result = parse_json_from_response(message.content[0].text)
    return {"success": True, "data": result}
