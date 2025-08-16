import os
import json
from pathlib import Path
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
# Future enhancement: Could add sentence transformers for semantic analysis
# from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool

from Classes.agent_state import AgentState


def _format_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


def _load_segments() -> List[Dict[str, Any]]:
    seg_path = Path("transcription_segments.json")
    if seg_path.exists():
        data = json.loads(seg_path.read_text(encoding="utf-8"))
        return data.get("segments", [])
    # Fallback: derive one pseudo-segment from transcription.txt
    txt_path = Path("transcription.txt")
    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8").strip()
        if text:
            return [{"start": 0.0, "end": 0.0, "start_str": "00:00", "end_str": "00:00", "text": text}]
    return []


@tool
def recommend_clips(state: AgentState):
    """
    Analyze Whisper transcription segments to recommend 15–60s sermon clips.

    Produces recommendations.json and recommendations.txt with entries like:
    Start: MM:SS, End: MM:SS, Description: "...", Confidence: 0.0–1.0

    Returns a JSON string with {"count", "path_json", "path_txt"}.
    """
    segments = _load_segments()
    if not segments:
        raise RuntimeError("No transcription found. Run transcribe_audio first.")

    # Prepare compact context for the LLM (limit size for cost/latency)
    # We keep up to ~150 segments or ~12k chars, whichever smaller
    compact: List[Dict[str, Any]] = []
    char_budget = 12000
    total_chars = 0

    # Instead of using words we could use embeddings to filter out noise
    # Future enhancement: Add sentence transformers for semantic analysis
    #
    # Potential noise filtering with embeddings:
    # 1. Calculate semantic similarity to "sermon content" vs "background noise"
    # 2. Filter out segments with low content quality scores
    # 3. Remove repetitive worship/music transcription errors
    # 4. Identify and skip technical setup discussions
    #
    # Example implementation:
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    #
    # def filter_noise_segments(segments):
    #     sermon_ref = "biblical teaching theological discussion scripture"
    #     noise_ref = "microphone sound check background music amen repetition"
    #
    #     filtered = []
    #     for segment in segments:
    #         text_embedding = model.encode(segment['text'])
    #         sermon_sim = cosine_similarity(text_embedding, model.encode(sermon_ref))
    #         noise_sim = cosine_similarity(text_embedding, model.encode(noise_ref))
    #
    #         if sermon_sim > noise_sim and len(segment['text'].split()) > 10:
    #             filtered.append(segment)
    #
    #     return filtered
    for seg in segments:
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        entry = {
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "start_str": seg.get("start_str") or _format_ts(float(seg.get("start", 0.0))),
            "end_str": seg.get("end_str") or _format_ts(float(seg.get("end", 0.0))),
            "text": txt,
        }
        # rough filtering to avoid very early pre-service noise based on keywords
        # actual filtering logic is left to the LLM instruction below
        compact.append(entry)
        total_chars += len(txt)
        if len(compact) >= 150 or total_chars >= char_budget:
            break

    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.3)

    system = (
        "You are an assistant that selects the best sermon clip segments for viral social media moments for a Church sermon. "
        "Use the provided transcript segments with timestamps to output recommended clips that are most likely to go viral. "
        "Follow these rules strictly: "
        "- Ignore pre-service chatter and background music as much as possible. Begin analysis when greetings like 'good morning', 'welcome', or the start of scripture discussion appear. "
        "- Target 60-90 seconds per clip (never exceed 180 seconds). "
        "- Each recommendation must be a clear, standalone message that works without extra context. "
        ""
        "CONTENT DIVERSITY REQUIREMENTS: "
        "- Select clips from different sermon segments: opening illustrations, main teaching points, practical applications, worship moments, personal stories, and closing prayers. "
        "- Avoid focusing on only one theological concept - diversify across multiple themes and sermon sections. "
        "- Include a mix of contemplative teaching moments, inspiring declarations, challenging questions, and uplifting conclusions. "
        ""
        "VOCAL EMPHASIS & ENGAGEMENT INDICATORS: "
        "- Actively prioritize moments with exclamations, imperative phrases, raised voice, passionate delivery, laughter, or emotional inflection. "
        "- Look for phrases like 'Listen!', 'Here's the thing...', 'You know what?', rhetorical questions, or emphatic statements. "
        "- Target segments where the pastor's energy and engagement are clearly elevated. "
        ""
        "RELATABILITY & ACCESSIBILITY: "
        "- Prioritize segments with contemporary references, pop culture connections, sports analogies, or practical life applications. "
        "- Look for stories that make theological concepts accessible to non-church audiences. "
        "- Include moments that demonstrate the pastor's personality: humor, vulnerability, storytelling ability, or conversational tone. "
        "- Balance profound biblical truths with language and examples that general audiences can understand and find compelling. "
        ""
        "EMOTIONAL RANGE & VIRAL POTENTIAL: "
        "- Select clips that evoke strong emotional responses: inspiration, conviction, hope, laughter, or deep reflection. "
        "- Include moments that are highly quotable, shareable, or likely to spark meaningful conversations. "
        "- Consider how each clip would perform across different social media contexts (Instagram, TikTok, YouTube Shorts). "
        "- Provide 3–8 strong recommendations if possible. "
        "- Base time ranges on the provided segment timestamps; you may merge consecutive segments to hit the target duration. "
        "- Return strict JSON only."
    )

    user = (
        "Transcript segments (JSON). Each item has start,end (sec), start_str,end_str (MM:SS), and text.\n\n"
        + json.dumps(compact, ensure_ascii=False) +
        "\n\nReturn a JSON object with key 'recommendations' containing an array. Each item must be: "
        "{ 'start_sec': number, 'end_sec': number, 'start': 'MM:SS', 'end': 'MM:SS', "
        "  'description': string, 'confidence': number (0-1), 'reasoning': string }. "
        "Do not include anything else besides this JSON."
    )

    resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])

    # Parse JSON safely
    try:
        data = json.loads(resp.content)
        recs = data.get("recommendations", [])
    except Exception:
        # Fallback empty
        data = {"recommendations": []}
        recs = []

    # Write outputs
    out_json = Path("recommendations.json")
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    for r in recs:
        start = r.get("start") or _format_ts(float(r.get("start_sec", 0)))
        end = r.get("end") or _format_ts(float(r.get("end_sec", 0)))
        desc = (r.get("description") or "").strip()
        conf = r.get("confidence")
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else ""
        lines.append(f"Start: {start}, End: {end}, Description: \"{desc}\"" + (f", Confidence: {conf_str}" if conf_str else ""))

    out_txt = Path("recommendations.txt")
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    return json.dumps({
        "count": len(recs),
        "path_json": str(out_json.resolve()),
        "path_txt": str(out_txt.resolve()),
    })

