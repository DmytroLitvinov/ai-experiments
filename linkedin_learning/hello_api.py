from openai import OpenAI  # version 1.0+

llm = OpenAI(
    api_key="dummy_for_lm_strudio_to_work",
    base_url="http://localhost:1234/v1",  # using LM Studio for local development
)

system_prompt = """Given the following short description
    of a particular topic, write 3 attention-grabbing headlines 
    for a blog post. Reply with only the titles, one on each line,
    with no additional text.
    DESCRIPTION:
"""
user_input = """AI Orchestration with LangChain and LlamaIndex
    keywords: Generative AI, applications, LLM, chatbot"""

response = llm.chat.completions.create(
    model="gpt-4-1106-preview",  # needed param to work on. it will not use OpenAI model at that case
    max_tokens=500,
    temperature=0.7,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ],
)

print(response.choices[0].message.content)
