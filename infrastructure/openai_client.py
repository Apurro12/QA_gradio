import openai

class OpenAIClient:
    def __init__(self):
        openai.api_key = "your-openai-api-key"

    def generate(self, prompt: str) -> str:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()