import litellm
from dotenv import load_dotenv

def _call_lm(model_name, prompt) -> str:
    load_dotenv()
    completion = litellm.completion(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content or ""


if __name__ == "__main__":
    model_name = "gemini/gemini-3-pro-preview"
    prompt = "Hello, how are you today?"
    response = _call_lm(model_name, prompt)
    print("LLM Response:")
    print(response)