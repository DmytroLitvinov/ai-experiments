import click
from crewai import Agent, Task, Crew, Process


def run():
    TOPIC = (
        "Local LLMs: how to choose between Ollama, vLLM and LM Studio (for beginners)"
    )

    writer = Agent(
        role="Writer",
        goal="Write concise, technically accurate posts for blogpost at website",
        backstory=(
            "You're a technical writer and developer. "
            "Write short, no water, give minimally sufficient code/arguments."
        ),
        verbose=True,
    )

    task = Task(
        description=f"Do a website blogpost on the topic: {TOPIC}",
        expected_output=(
            "Structured post up to 2500 characters, with TL;DR (3 bullet points), "
            "and a short conclusion. No mention of CrewAI."
        ),
        agent=writer,
    )

    crew = Crew(agents=[writer], tasks=[task], process=Process.sequential, verbose=True)
    print(crew.kickoff())
    # click.echo(click.style("Output: ", fg="yellow") + crew.kickoff())


if __name__ == "__main__":
    run()
