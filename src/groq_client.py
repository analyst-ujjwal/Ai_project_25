import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def run_on_groq(payload: dict):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set — falling back to local inference")

    # Placeholder fake response for demonstration
    predictions = [42.0] * len(payload.get("inputs", []))
    return {"predictions": predictions}
