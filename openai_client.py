from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Already loads it from `OPENAI_API_KEY` env variable
client = OpenAI()
