import os
import requests

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

def search_google(query, num_results=3):
    """
    Perform a Google Custom Search and return top snippet results.
    """
    if not GOOGLE_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
        raise EnvironmentError("Missing GOOGLE_API_KEY or GOOGLE_SEARCH_ENGINE_ID in environment.")

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_SEARCH_ENGINE_ID,
        "q": query,
        "num": num_results
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("items", [])

        snippets = []
        for item in results:
            title = item.get("title", "No Title")
            link = item.get("link", "No Link")
            snippet = item.get("snippet", "No Snippet")
            snippets.append(f"{title}\n{snippet}\n{link}")

        return "\n\n".join(snippets)

    except Exception as e:
        print(f"❌ Google search failed: {e}")
        return "Live search failed or unavailable at this time."
