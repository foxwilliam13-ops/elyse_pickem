"""Phase 2: short news notes per team, written by Gemini from Google News headlines.

How it stays honest:
- Gemini only sees a numbered list of real headlines from Google News RSS.
- Each note must say which headline numbers it came from. Notes that cite
  nothing (or cite numbers that don't exist) are thrown away.
- Links on the page come straight from the RSS feed, never from Gemini.
- No headlines, no key, or any error -> the team simply gets no AI notes
  and the page says "Nothing notable".

Needs GEMINI_API_KEY (free tier, from Google AI Studio). Optional: GEMINI_MODEL.
Standard library only.
"""

import email.utils
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NEWS_RSS = "https://news.google.com/rss/search"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.5-flash-lite"
MAX_HEADLINES = 10      # headlines sent to Gemini per team
MAX_NOTES = 3           # notes kept per team
DAYS_BACK = 7
GEMINI_GAP = 5.0        # seconds between Gemini calls (keeps us under the free per-minute limit)
NEWS_GAP = 1.0          # seconds between Google News fetches (be polite)

PROMPT = """You help someone research a college football game. Below are recent news headlines about {team} football, numbered.

Write at most {max_notes} short notes (one sentence each, under 25 words) about things that could matter for this week's game: injuries, suspensions, quarterback or coaching changes, players opting out or transferring, team turmoil, travel or weather problems.

Rules:
- Use ONLY facts stated in the headlines. Do not add anything from your own knowledge.
- Every note must list the headline numbers it is based on.
- Skip game recaps, rankings talk, betting picks, recruiting, and anything about other sports.
- Never say who will win or cover. Do not recommend a pick.
- If nothing is relevant, return an empty list. That is a good answer.

Return JSON only: {{"notes": [{{"text": "...", "headlines": [1, 3]}}]}}

Headlines:
{headlines}"""

calls = {"news": 0, "gemini": 0}


def _get(url, headers=None, data=None, timeout=30):
    request = urllib.request.Request(url, data=data, headers={"User-Agent": "elyse-pickem-dashboard", **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def headlines_for(query):
    """Recent Google News headlines for a search: [{title, source, url, date}]."""
    params = {"q": f'"{query}" football when:{DAYS_BACK}d', "hl": "en-US", "gl": "US", "ceid": "US:en"}
    calls["news"] += 1
    root = ET.fromstring(_get(f"{NEWS_RSS}?{urllib.parse.urlencode(params)}"))
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].strip()
        link = (item.findtext("link") or "").strip()
        if not title or not link.startswith("https://"):
            continue
        try:
            when = email.utils.parsedate_to_datetime(item.findtext("pubDate") or "")
        except (TypeError, ValueError):
            when = None
        items.append({"title": title, "source": source, "url": link, "date": when})
    items.sort(key=lambda h: h["date"].timestamp() if h["date"] else 0, reverse=True)
    return items[:MAX_HEADLINES]


def ask_gemini(team, headlines, key, model):
    listing = "\n".join(
        f"{i}. {h['title']} ({h['source'] or 'unknown source'}"
        + (f", {h['date']:%b} {h['date'].day}" if h["date"] else "") + ")"
        for i, h in enumerate(headlines, 1))
    body = {
        "contents": [{"parts": [{"text": PROMPT.format(team=team, max_notes=MAX_NOTES, headlines=listing)}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }
    url = f"{GEMINI_BASE}/{urllib.parse.quote(model)}:generateContent"
    calls["gemini"] += 1
    raw = _get(url, {"Content-Type": "application/json", "x-goog-api-key": key}, json.dumps(body).encode(), timeout=60)
    reply = json.loads(raw)
    text = "".join(p.get("text", "") for p in reply["candidates"][0]["content"]["parts"])
    return json.loads(text).get("notes") or []


def checked_notes(raw_notes, headlines):
    """Keep only notes that cite real headline numbers; attach those headlines' links."""
    out = []
    for note in raw_notes:
        if not isinstance(note, dict):
            continue
        text = str(note.get("text") or "").strip()
        cited = []
        for n in note.get("headlines") or []:
            if isinstance(n, int) and 1 <= n <= len(headlines) and n not in cited:
                cited.append(n)
        if not text or not cited:
            continue
        out.append({
            "text": text[:300],
            "sources": [{"title": headlines[n - 1]["title"], "source": headlines[n - 1]["source"],
                         "url": headlines[n - 1]["url"]} for n in cited],
        })
        if len(out) == MAX_NOTES:
            break
    return out


def team_news(teams):
    """teams: {display name: search name}. Returns {display name: [notes]} for teams we managed to check.

    A team missing from the result means "not checked" (no key or an error), so the
    page can tell that apart from "checked, nothing notable" (an empty list)."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print("  news notes skipped: no GEMINI_API_KEY")
        return {}
    model = os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
    out = {}
    for name, search in teams.items():
        try:
            time.sleep(NEWS_GAP)
            headlines = headlines_for(search)
        except Exception as err:
            print(f"  news skipped for {name}: {err}")
            continue
        if not headlines:
            out[name] = []
            continue
        for attempt in (1, 2):
            try:
                time.sleep(GEMINI_GAP)
                out[name] = checked_notes(ask_gemini(search, headlines, key, model), headlines)
                break
            except urllib.error.HTTPError as err:
                if err.code == 429 and attempt == 1:
                    print("  Gemini rate limit hit, waiting a minute")
                    time.sleep(60)
                    continue
                if err.code in (400, 401, 403, 404):
                    print(f"  Gemini refused the request ({err.code}). Check GEMINI_API_KEY and GEMINI_MODEL. Stopping news notes.")
                    return out
                print(f"  Gemini failed for {name}: {err}")
                break
            except Exception as err:
                print(f"  Gemini failed for {name}: {err}")
                break
    print(f"  news notes: {len(out)} of {len(teams)} teams checked "
          f"({calls['news']} news fetches, {calls['gemini']} Gemini calls)")
    return out
