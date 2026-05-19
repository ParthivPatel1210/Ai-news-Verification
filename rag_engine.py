import os
import json
import google.generativeai as genai
from duckduckgo_search import DDGS
from urllib.parse import urlparse

# Check if Gemini API is configured
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Using Gemini 1.5 Flash as it is fast and cheap for this task
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None

def is_available():
    return model is not None

def search_web_verification(text):
    try:
        words = text.split()
        if len(words) > 10:
            headline = ' '.join(words[:10]) + " news fact check"
        else:
            headline = ' '.join(words) + " news fact check"
            
        results = DDGS().text(headline, max_results=3)
        out = []
        if results:
            for r in results:
                out.append({
                    'title': r.get('title', ''),
                    'href': r.get('href', ''),
                    'body': r.get('body', '')[:200] + '...'
                })
        return out
    except Exception as e:
        print(f"Web Search Error: {e}")
        return []

def verify_news_with_rag(text, url=None):
    """
    Returns (prediction, probability, explanation, web_results)
    """
    if not is_available():
        return None, None, "Gemini API key not configured.", []
        
    web_results = search_web_verification(text)
    
    context = ""
    for idx, res in enumerate(web_results):
        context += f"Source {idx+1}: {res['title']}\nURL: {res['href']}\nSnippet: {res['body']}\n\n"
        
    prompt = f"""
    You are an expert fact-checker and AI news verification bot.
    Analyze the following news text and determine its authenticity based on the provided web search context.

    News Text: "{text}"
    Source URL (if any): {url if url else 'None'}

    Web Search Context:
    {context if context else 'No web search results found.'}

    Output your analysis strictly in JSON format with the following keys:
    "prediction": "REAL" or "FAKE",
    "probability": A float between 0.0 and 1.0 representing your confidence that the news is REAL (1.0 = extremely confident it's real, 0.0 = extremely confident it's fake),
    "explanation": A detailed, human-readable paragraph explaining your verdict. Cite the 'Source X' from the context if it helped. If the text is a known conspiracy or satire, state that. 

    Do NOT include Markdown formatting in your response (like ```json), just return the raw JSON object.
    """
    
    try:
        response = model.generate_content(prompt)
        # Parse JSON
        resp_text = response.text.strip()
        if resp_text.startswith("```json"):
            resp_text = resp_text[7:]
        if resp_text.endswith("```"):
            resp_text = resp_text[:-3]
            
        result = json.loads(resp_text)
        
        # formatting the explanation to HTML
        explanation = result.get('explanation', '')
        
        return result.get('prediction', 'FAKE'), result.get('probability', 0.5), explanation, web_results
        
    except Exception as e:
        print(f"RAG Engine Error: {e}")
        return None, None, f"Error generating verification: {str(e)}", web_results
