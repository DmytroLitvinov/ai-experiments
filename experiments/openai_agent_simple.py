"""
Based on article https://www.dolthub.com/blog/2025-09-19-agentic-development/
"""
import asyncio
from agents import Agent, Runner, SQLiteSession, function_tool
from dotenv import load_dotenv

load_dotenv()


@function_tool
def read_file(file_path:str, start:int = 0, num_bytes:int = -1) -> bytes:
    """
    Reads a portion of a local file from a specified start position for a given number of bytes.

    Args:
        file_path (str): The path to the local file.
        start (int): The starting byte position to read from.
        num_bytes (int): The number of bytes to read. If -1, reads until the end of the file.

    Returns:
        bytes: The binary content read from the file
    """
    with open(file_path, 'rb') as file:
        file.seek(start)

        if num_bytes == -1:
            content = file.read()
        else:
            content = file.read(num_bytes)

        return content


async def session_conversation():
    session = SQLiteSession("conversation_name")
    agent = Agent(name="Assistant", instructions="Answer whatever questions I ask you.")
    result = await Runner.run(agent, "Generate a random number between 1 and 100.", session=session)
    print(result.final_output)

    result = await Runner.run(agent, "Now add 10 to the number you gave me.", session=session)
    print(result.final_output)

async def tool_conversation():
    agent = Agent(name="Assistant", instructions="Use your ability to read files from the local filesystem to answer questions.", tools=[read_file])
    result = await Runner.run(agent, "What is the sum of columns w,x,y, and z in the file tmp/test.csv?")
    print(result.final_output)


if __name__ == "__main__":
    # asyncio.run(session_conversation())
    asyncio.run(tool_conversation())