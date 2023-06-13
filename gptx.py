#!/usr/bin/python3
import json
import random
import re
import string
import sys
from pathlib import Path
from typing import Iterator, List, Optional, TypedDict, Tuple, Dict

import click
import openai
import requests
from typing_extensions import Never


class Message(TypedDict):
    role: str
    content: str


Prompt = List[Message]


DEFAULT_MODEL = "gpt-4"
WORKDIR = Path.home() / ".gptx"
CONV_DIR = WORKDIR / "conversations"
LATEST_CONV_FILE = CONV_DIR / "latest.txt"
PROMPT_DIR = WORKDIR / "prompts"
API_KEY_FILE = WORKDIR / "api-key.txt"


DEFAULT_PROMPTS: Dict[str, Prompt] = dict(
default=[Message(role="system", content="""\
You are a useful AI assistant that runs on the terminal.
Your answers are highly professional and to the point.\
""")],
bash=[Message(role="system", content="""\
You are an AI writing Bash commands running directly in the terminal.
You output raw, expertly written Bash commands, NO MARKUP (```).\
""")],
)


printerr = lambda *args, **kwargs: print(*args, file=sys.stderr, **kwargs)


def fail(msg: str) -> Never:
    printerr(msg)
    exit(1)


def resolve_conversation_id(conversation_id: str) -> str:
    if conversation_id.strip().lower() == "latest":
        latest = get_latest_conversation_id()
        if latest is None:
            fail("Latest conversation not found.")
        conversation_id = latest
    return conversation_id


def get_conversation_path(conversation_id: str) -> Path:
    conversation_id = resolve_conversation_id(conversation_id)
    path = CONV_DIR / f"{conversation_id}.json"
    return path


def get_prompt_path(prompt_id: str) -> Path:
    path = PROMPT_DIR / f"{prompt_id}.json"
    return path


def load_prompt(prompt_id: str) -> Prompt:
    bootstrap_default_prompts()
    path = get_prompt_path(prompt_id)
    if not path.exists():
        fail(f"Prompt not found: {prompt_id}")
    return json.loads(path.read_text())


def bootstrap_default_prompts() -> str:
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    for prompt_id, prompt in DEFAULT_PROMPTS.items():
        path = get_prompt_path(prompt_id)
        if path.exists():
            continue
        path.write_text(json.dumps(prompt))
    return "default"


def get_latest_conversation_id() -> Optional[str]:
    if not LATEST_CONV_FILE.exists():
        return None
    return LATEST_CONV_FILE.read_text().strip()


def load_or_create_conversation(
    conversation_id: str,
    prompt_id: str,
) -> List[Message]:
    path = get_conversation_path(conversation_id)
    if not path.exists():
        prompt = load_prompt(prompt_id)
        return list(prompt)
    return json.loads(path.read_text())


def load_conversation(conversation_id: str) -> List[Message]:
    path = get_conversation_path(conversation_id)
    if not path.exists():
        fail(f"Conversation not found: {conversation_id}")
    return json.loads(path.read_text())


def save_conversation(conversation_id: str, messages: List[Message]) -> None:
    conversation_id = resolve_conversation_id(conversation_id)
    path = get_conversation_path(conversation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(messages, f)
    LATEST_CONV_FILE.write_text(conversation_id)


def next_conversation_id() -> str:
    pool = string.ascii_lowercase + string.digits
    ATTEMPTS = 10_000
    for _ in range(ATTEMPTS):
        if not get_conversation_path(conversation_id := "".join(random.choices(pool, k=3))).exists():
            return conversation_id
    fail(f"Failed to generate a conversation ID after {ATTEMPTS:,} attempts.")


def get_conversation_ids() -> List[str]:
    return [path.stem for path in CONV_DIR.glob("*.json")]


def enhance_content(
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
    messages: List[Message],
    api_key: str,
    max_tokens: int,
    temperature: float,
    model: str,
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
@click.option("--prompt", "-p", type=str, help="Prompt ID", default="default")
@click.option("--model", type=str, help="Model", default=DEFAULT_MODEL)
@click.option("--max-length", type=int, help="Max chars in prompt", default=2 ** 13)
@click.argument("user_message", nargs=-1, required=True)
# fmt: on
def query(
    max_tokens: int,
    temperature: float,
    api_key_file: Path,
    conversation: str,
    prompt: str,
    model: str,
    user_message: List[str],
    max_length: int,
) -> None:
    """Query GPT4"""
    api_key = api_key_file.read_text().strip()
    conversation_id = conversation or next_conversation_id()
    conversation_id = resolve_conversation_id(conversation_id)
    prompt_id = prompt
    messages = load_or_create_conversation(conversation_id, prompt_id)
    message_str = enhance_content(" ".join(user_message).strip())
    if not message_str:
        fail("Empty message.")
    if len(message_str) > max_length:
        fail(
            f"Message is too long: {len(message_str):,} characters. Set --max-length to override."
        )
    messages.append(Message(role="user", content=message_str))
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
    messages.append(Message(role="assistant", content=full_answer))
    save_conversation(conversation_id, messages)


@cli.command("new-prompt")
@click.option("--prompt", "-p", type=str, help="Prompt ID", required=True)
@click.option("--message", "-m", type=str, multiple=True,
              help=("Message to add to prompt. First word is the role (user or assistant), "
                    "followed by content"))
def new_prompt(
    prompt: str,
    message: Tuple[str, ...],
) -> None:
    role_content_pairs = [m.lstrip().split(" ", maxsplit=1) for m in message]
    for role, content in role_content_pairs:
        if role not in {"user", "assistant", "system"}:
            fail(f"Invalid role: {role}, expected 'user', 'assistant' or 'system'")
        if not content.strip():
            fail(f"Empty content for role: {role}")
    messages: Prompt = [Message(role=role, content=content)
                        for role, content in role_content_pairs]
    path = get_prompt_path(prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(messages, indent=2))


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
    conversation_id = resolve_conversation_id(conversation_id)
    path = get_conversation_path(conversation_id)
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
