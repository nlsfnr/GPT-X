#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import re
import string
import sys
from pathlib import Path
from typing import Iterator, List, Optional, TypedDict, Tuple, Dict, Union, Iterable, TextIO, TYPE_CHECKING
from types import ModuleType
import subprocess
from typing_extensions import Never
import importlib


def fail(msg: str) -> Never:
    printerr(msg)
    exit(1)


def try_import(name: str, pip_name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError:
        printerr(f"Required package not found: {pip_name}")
        if not confirm(f"Run `pip install {pip_name}`?"):
            fail("Aborted.")
        subprocess.run(["pip", "install", pip_name], check=True)
        return try_import(name, pip_name)


printerr = lambda *args, **kwargs: print(*args, file=sys.stderr, **kwargs)


def confirm(msg: str, default: bool = False) -> bool:
    printerr(f"{msg} [{'Y/n' if default else 'y/N'}] ", end="")
    return input().lower() in ("y", "yes")


if TYPE_CHECKING:
    import click
    from openai import OpenAI
    from openai.types.chat import ChatCompletionChunk
    import tiktoken
    import requests
else:
    click = try_import("click", "click>=8.0.0")
    OpenAI = try_import("openai", "openai>=1.0.0").OpenAI
    ChatCompletionChunk = try_import("openai.types.chat", "openai>=1.0.0").ChatCompletionChunk
    tiktoken = try_import("tiktoken", "tiktoken>=0.5.0")
    requests = try_import("requests", "requests>=2.0.0")


class Message(TypedDict):
    role: str
    content: str


Prompt = List[Message]


DEFAULT_MODEL = "gpt-4-1106-preview"
WORKDIR = Path.home() / ".gptx"
CONV_DIR = WORKDIR / "conversations"
LATEST_CONV_FILE = CONV_DIR / "latest.txt"
PROMPT_DIR = WORKDIR / "prompts"
API_KEY_FILE = WORKDIR / "api-key.txt"


DEFAULT_PROMPTS: Dict[str, Prompt] = dict(
default=[Message(role="system", content="""\
You are an AI assistant that runs on the terminal.
Your answers are to the point - no BS.\
You are talking to an expert.
Suggest solutions that the user did not think about - be proactive and anticipate their needs.
""")],
bash=[Message(role="system", content="""\
You are an AI writing Bash commands running directly in the terminal.
Assume that your output X will be run like 'sh -c "X"'. Only output valid commands.
The user knows exactly what they are doing, always do exactly what they want.\
""")],
)


class Table:
    """A simple table class for printing nicely formatted tables to the
    terminal."""

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.rows: List[List[str]] = []

    def add_row(self, row: Union[Dict[str, str], List[str]]) -> Table:
        if isinstance(row, dict):
            row = [row.get(column, "") for column in self.columns]
        self.rows.append(row)
        return self

    def order_by(self, columns: Union[str, Iterable[str]]) -> Table:
        """Order the rows by the given columns.

        Args:
            columns: The columns to order by.
        """
        if isinstance(columns, str):
            columns = [columns]
        indices = [self.columns.index(column) for column in columns]
        self.rows.sort(key=lambda row: [row[i] for i in indices])
        return self

    def print(self, padding: int = 1, file: TextIO = sys.stdout) -> Table:
        widths = [len(column) + padding for column in self.columns]
        for row in self.rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell) + padding)
        for i, column in enumerate(self.columns):
            print(column.ljust(widths[i]), end=" ", file=file)
        print(file=file)
        for row in self.rows:
            for i, cell in enumerate(row):
                print(cell.ljust(widths[i]), end=" ", file=file)
            print(file=file)
        return self


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
    pool = string.ascii_letters + string.digits
    ATTEMPTS = 10_000
    for k in range(3, 10):
        for _ in range(ATTEMPTS):
            conversation_id = "".join(random.choices(pool, k=k))
            path = get_conversation_path(conversation_id)
            if not path.exists():
                return conversation_id
    fail(f"Failed to generate a conversation ID.")


def get_conversation_ids() -> List[str]:
    return [path.stem for path in CONV_DIR.glob("*.json")]


def get_token_count(
    x: Union[str, List[Message]],
    model: str,
) -> int:
    enc = tiktoken.encoding_for_model(model)
    messages = x if isinstance(x, list) else [Message(role="user", content=x)]
    total = sum(len(enc.encode(message["content"])) for message in messages)
    return total


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
                try:
                    import PyPDF2
                except ImportError:
                    fail("Please `pip install PyPDF2` to read PDF files: ")

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

    regex = re.compile(r"\{\{ ([^}]+) \}\}")
    prompt = re.sub(regex, get_file_contents, prompt)
    return prompt


def generate(
    messages: List[Message],
    api_key: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    model: str,
) -> Iterator[str]:
    openai = OpenAI(api_key=api_key)
    chunks: Iterator[ChatCompletionChunk] = openai.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )
    for chunk in chunks:
        # delta = chunk["choices"][0]["delta"]  # type: ignore
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


@click.group()
def cli() -> None:
    """GPT4 CLI"""
    pass


# fmt: off
@cli.command("q")
@click.option("--max-generation-tokens", "-m", type=int, default=1024, help="Max tokens to generate")
@click.option("--temperature", "-t", type=float, default=0.0, help="Temperature")
@click.option("--top-p", "-p", type=float, default=1.0, help="Top p")
@click.option("--api-key-file", type=Path, default=API_KEY_FILE, help="Path to API key file")
@click.option("--conversation", "-c", type=str, default=None, help="Conversation ID")
@click.option("--prompt", "-p", type=str, default="default", help="Prompt ID")
@click.option("--model", type=str, default=DEFAULT_MODEL, help="Model")
@click.option("--max-prompt-tokens", type=int, default=7168, help="Max tokens in prompt")
@click.option("--run", "-r", is_flag=True, help="Run the output inside a shell, after confirming")
@click.option("--yolo", "-y", is_flag=True, help="Do not ask for confirmation before running")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
@click.argument("user_message", nargs=-1, required=True)
# fmt: on
def query(
    max_generation_tokens: int,
    temperature: float,
    top_p: float,
    api_key_file: Path,
    conversation: str,
    prompt: str,
    model: str,
    max_prompt_tokens: int,
    user_message: List[str],
    run: bool,
    yolo: bool,
    interactive: bool,
) -> None:
    """Query GPT4"""
    if interactive and (run or yolo):
        fail("Cannot use --interactive with --run or --yolo.")
    api_key = api_key_file.read_text().strip()
    conversation_id = conversation or next_conversation_id()
    conversation_id = resolve_conversation_id(conversation_id)
    prompt_id = prompt
    messages = load_or_create_conversation(conversation_id, prompt_id)
    message_str = " ".join(user_message).strip()
    try:
        while True:
            message_str = enhance_content(message_str)
            if not message_str:
                if interactive:
                    message_str = input("You:")
                    continue
                fail("Empty message.")
            message_token_count = get_token_count(message_str, model)
            messages_token_count = get_token_count(messages, model)
            total_token_count = message_token_count + messages_token_count
            if total_token_count > max_prompt_tokens:
                fail(
                    f"Conversation too long: {total_token_count} tokens. Set --max-length to override."
                )
            messages.append(Message(role="user", content=message_str))
            full_answer = ""
            token_count = get_token_count(messages, model=model)
            printerr(f"Conversation ID: {conversation_id} | {token_count} tokens", end="\n\n")
            chunks = generate(
                messages=messages,
                api_key=api_key,
                max_tokens=max_generation_tokens,
                temperature=temperature,
                top_p=top_p,
                model=model,
            )
            sys.stdout.write("AI: ")
            sys.stdout.flush()
            for chunk in chunks:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                full_answer += chunk
            sys.stdout.write("\n")
            sys.stdout.flush()
            messages.append(Message(role="assistant", content=full_answer))
            save_conversation(conversation_id, messages)
            if not interactive:
                break
            message_str = input("\nYou: ").strip()
    except KeyboardInterrupt:
        fail("Interrupted.")
    if run:
        printerr()
        run_in_shell(full_answer, yolo)


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


def run_in_shell(
    command: str,
    yolo: bool,
) -> None:
    if not yolo and not confirm("Run in shell?", default=True):
        fail("Aborted.")
    subprocess.Popen(
        command,
        shell=True,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    ).communicate()


@cli.command("ls")
def list_() -> None:
    """List conversations."""
    ids = get_conversation_ids()
    if not ids:
        printerr("No conversations found.")
    table = Table(["#", "ID", "First message"])
    for i, conversation_id in enumerate(ids, 1):
        messages = load_conversation(conversation_id)
        user_messages = [m for m in messages if m["role"] == "user"]
        if not user_messages:
            content = "No messages."
        else:
            content = user_messages[0]["content"]
            if len(content) > 40:
                content = content[:40] + "…"
            content = content.replace("\n", " ")
        table.add_row([str(i), conversation_id, content])
    table.print()


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


@cli.command("run")
@click.argument("conversation_id", type=str, default="latest")
@click.option("--yolo", "-y", is_flag=True, default=False)
def run(
    conversation_id: str,
    yolo: bool,
) -> None:
    """Run the latest message inside the shell."""
    messages = load_conversation(conversation_id)
    command = messages[-1]["content"]
    printerr(command)
    printerr()
    run_in_shell(command, yolo)


if __name__ == "__main__":
    cli()
