from experiments.system_prompts import useful_assistant_prompt
from experiments.constants import MODEL_ENGINE
from openai_client import client
import click

history = [
    {
        "role": "system",
        "content": useful_assistant_prompt,
    }
]

while True:
    request: str = input(
        click.style('Input: (enter "exit"/"quit" to exit the program): ', fg="green")
    )

    if request.lower() in ("exit", "quit"):
        break

    # Add message to history
    history.append(
        {
            "role": "user",
            "content": request,
        }
    )

    response = client.chat.completions.create(
        model=MODEL_ENGINE,
        messages=history,
    )

    content = response.choices[0].message.content.strip()

    # Debug output
    click.echo(click.style("History: ", fg="blue") + str(history))

    # Output to user
    click.echo(click.style("Output: ", fg="yellow") + content)

    history.append(
        {
            "role": "assistant",
            "content": content,
        }
    )

    click.echo()
