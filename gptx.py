#!/usr/bin/python3
import json
import random
import re
import string
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import click
import openai
import requests
from typing_extensions import Never

SYSTEM_PROMPT = """\
You are a useful AI assistant that runs on the terminal. \
Your answers are brief and to the point.\
"""
MODEL = "gpt-4"
WORKDIR = Path.home() / ".gptx" / "conversations/"
API_KEY_FILE = Path.home() / ".gptx" / "api-key.txt"


printerr = lambda *args, **kwargs: print(*args, file=sys.stderr, **kwargs)


def fail(msg: str) -> Never:
    printerr(msg)
    exit(1)


def get_path(conversation_id: str) -> Path:
    if conversation_id.strip().lower() == "latest":
        latest = get_latest_conversation_id()
        if latest is None:
            fail("Latest conversation not found.")
        conversation_id = latest
    path = WORKDIR / f"{conversation_id}.json"
    return path


def get_latest_conversation_id() -> Optional[str]:
    latest = WORKDIR / "latest.txt"
    if not latest.exists():
        return None
    return latest.read_text().strip()


def load_or_create_conversation(conversation_id: str) -> List[Dict[str, str]]:
    path = get_path(conversation_id)
    if not path.exists():
        return [dict(role="system", content=SYSTEM_PROMPT)]
    return json.loads(path.read_text())


def load_conversation(conversation_id: str) -> List[Dict[str, str]]:
    path = get_path(conversation_id)
    if not path.exists():
        fail(f"Conversation not found: {conversation_id}")
    return json.loads(path.read_text())


def save_conversation(conversation_id: str, messages: List[Dict[str, str]]) -> None:
    path = get_path(conversation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(messages, f)
    latest = WORKDIR / "latest.txt"
    latest.write_text(conversation_id)


def next_conversation_id() -> str:
    pool = string.ascii_lowercase + string.digits
    ATTEMPTS = 10_000
    for _ in range(ATTEMPTS):
        if not get_path(conversation_id := "".join(random.choices(pool, k=3))).exists():
            return conversation_id
    fail(f"Failed to generate a conversation ID after {ATTEMPTS:,} attempts.")


def get_conversation_ids() -> List[str]:
    return [path.stem for path in WORKDIR.glob("*.json")]


def enhance_prompt(
    prompt: str,
) -> str:
    def get_file_contents(match: re.Match) -> str:
        """Inject file contents into the prompt."""
        path_str = match.group(1)
        if path_str.startswith("http"):
            response = requests.get(path_str)
            response.raise_for_status()
            text = response.text
            printerr(f"Injecting: {path_str}\t{len(text)} chars")
        elif path_str == "stdin":
            text = sys.stdin.read()
            printerr(f"Injecting: stdin\t{len(text)} chars")
        else:
            path = Path(path_str)
            if not path.exists():
                fail(f"File not found: {path}")
            if path.suffix.lower() == ".pdf":
                import PyPDF2

                text = ""
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfFileReader(f)
                    for page in reader.pages:
                        text += page.extractText()
                return text
            else:
                text = path.read_text()
            printerr(f"Injecting: {path}\t{len(text)} chars")
        return text

    regex = r"\{\{ ([^}]+) \}\}"
    prompt = re.sub(regex, get_file_contents, prompt)
    return prompt


def generate(
    messages: List[Dict[str, str]],
    api_key: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
    model: str = MODEL,
) -> Iterator[str]:
    chunks = openai.ChatCompletion.create(
        model=model,
        api_key=api_key,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    for chunk in chunks:
        delta = chunk["choices"][0]["delta"]  # type: ignore
        if "content" in delta:
            yield delta["content"]


@click.group()
def cli() -> None:
    """GPT4 CLI"""
    pass


# fmt: off
@cli.command("q")
@click.option("--max-tokens", "-m", type=int, help="Max tokens to generate", default=128)
@click.option("--temperature", "-t", type=float, help="Temperature", default=0.0)
@click.option("--api-key-file", type=Path, help="Path to API key file", default=API_KEY_FILE)
@click.option("--conversation", "-c", type=str, help="Conversation ID", default=None)
@click.option("--model", type=str, help="Model", default=MODEL)
@click.option("--max-length", type=int, help="Max chars in prompt", default=2 ** 13)
@click.argument("prompt", nargs=-1, required=True)
# fmt: on
def query(
    max_tokens: int,
    temperature: float,
    api_key_file: Path,
    conversation: str,
    model: str,
    prompt: List[str],
    max_length: int,
) -> None:
    """Query GPT4"""
    api_key = api_key_file.read_text().strip()
    conversation_id = conversation or next_conversation_id()
    messages = load_or_create_conversation(conversation_id)
    prompt_str = enhance_prompt(" ".join(prompt).strip())
    if not prompt_str:
        fail("Empty prompt.")
    if len(prompt_str) > max_length:
        fail(
            f"Prompt is too long: {len(prompt_str):,} characters. Set --max-length to override."
        )
    messages.append(dict(role="user", content=prompt_str))
    full_answer = ""
    printerr(f"Conversation ID: {conversation_id}", end="\n\n")
    chunks = generate(
        messages=messages,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        model=model,
    )
    for chunk in chunks:
        print(chunk, end="")
        full_answer += chunk
    print()
    messages.append(dict(role="assistant", content=full_answer))
    save_conversation(conversation_id, messages)


@cli.command("ls")
def list_() -> None:
    """List conversations."""
    ids = get_conversation_ids()
    if not ids:
        printerr("No conversations found.")
    for i, conversation_id in enumerate(ids, 1):
        messages = load_conversation(conversation_id)
        user_messages = [m for m in messages if m["role"] == "user"]
        if not user_messages:
            content = "No messages."
        else:
            content = user_messages[0]["content"]
            if len(content) > 40:
                content = content[:40] + "..."
            content = content.replace("\n", " ")
        print(f"{i:2d}\t{conversation_id}\t{content}")


@cli.command("rm")
@click.argument("conversation_id", type=str, default="latest")
def remove(conversation_id: str) -> None:
    """Remove a conversation."""
    path = get_path(conversation_id)
    if not path.exists():
        fail(f"Conversation {conversation_id} not found.")
    path.unlink()
    printerr(f"Conversation {conversation_id} removed.")


@cli.command("print")
@click.argument("conversation_id", type=str, default="latest")
def print_(conversation_id: str) -> None:
    """Print a conversation."""
    messages = load_conversation(conversation_id)
    for message in messages:
        print(f"{message['role']}: {message['content']}")


@cli.command("repeat")
@click.argument("conversation_id", type=str, default="latest")
def repeat(conversation_id: str) -> None:
    """Repeat the latest message."""
    messages = load_conversation(conversation_id)
    last_message = messages[-1]
    print(f"{last_message['content']}")


if __name__ == "__main__":
    cli()
