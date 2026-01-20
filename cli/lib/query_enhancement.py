import os
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.
Only correct obvious typos. Don't change correctly spelled words.
Query: "{query}"
If no errors, return the original query.
Corrected:"""

    try:
        response = client.models.generate_content(model=model, contents=prompt)
        corrected = (response.text or "").strip().strip('"')
        return corrected if corrected else query
        
    except errors.ClientError as e:
        # Check specifically for Rate Limit / Quota errors
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print(f"\n[!] Spell-check quota exceeded. Using original query: '{query}'")
            return query
        # Re-raise other unexpected API errors (like auth or connection)
        raise e

def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""

    try:
        response = client.models.generate_content(model=model, contents=prompt)
        rewritten = (response.text or "").strip().strip('"')
        return rewritten if rewritten else query
    except errors.ClientError as e:
        # Check specifically for Rate Limit / Quota errors
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print(f"\n[!] Rewrite quota exceeded. Using original query: '{query}'")
            return query
        # Re-raise other unexpected API errors (like auth or connection)
        raise e
  

def expand_query(query: str) -> str:
    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""

    try:
        response = client.models.generate_content(model=model, contents=prompt)
        expanded_terms = (response.text or "").strip().strip('"')
        return f"{query} {expanded_terms}"
    except errors.ClientError as e:
        # Check specifically for Rate Limit / Quota errors
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print(f"\n[!] Expansion quota exceeded. Using original query: '{query}'")
            return query
        # Re-raise other unexpected API errors (like auth or connection)
        raise e
   


def enhance_query(query: str, method: Optional[str] = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite_query(query)
        case "expand":
            return expand_query(query)
        case _:
            return query
