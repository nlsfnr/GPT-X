# GPTX

Bringing the Unix-philosophy to ChatGPT.

GPTX is a Python module that allows you to access OpenAI's Chat-models from the
terminal. Apart from asking general questions you can e.g. generate bash
commands, inject file contents, websites and stdin into your prompts.

```bash
$ gptbash Grep for all python imports, not including .venv/ or .env/
...
$ gptx repeat| sh | gpt4 create a minimal requirements.txt for the following imports:\n{{ stdin }}
...
$ gpt4r add mypy, flake8, isort and black
Conversation ID: oni

requirements.txt:

click
openai
requests
PyPDF2
mypy
flake8
isort
black
```

## Installation

1. Clone the repository.
2. Install the required dependencies `pip install -r requirements.txt`
3. Paste your OpenAI API key into `~/.gptx/api-key.txt`
4. Add `gptx.py` to yout PATH, e.g. `ln -s "$(pwd)/gptx.py" ~/.local/bin/gptx` (assuming you are inside the repository)

I strongly recommend adding the following aliases to your `.bashrc` or `.bash_aliases`:

```bash
alias gpt4="gptx q --model gpt-4"  # To ask a question in a new conversation
alias gpt4r="gptx q --model gpt-4 --conversation latest"  # Continue the latest conversation
alias gptbash="gptx q --model gpt-4 --prompt bash"  # Output raw bash commands
```

## Usage

Note: `conversation_id` defaults to `latest` which is a special ID that
points to the latest conversation.

You can inject text into your prompt using `{{ my-file.txt }}`. Currently,
paths to (text-based) files, PDFs and URLs are supported.

```bash
gptx [COMMAND] [OPTIONS] [ARGUMENTS]
```

### Examples

```bash
gptx q What does this code do?\n\n{{ gptx.py }}
gptx q -m 1000 -c latest Write a README.md for the following Python module:\n\n{{ gptx.py }}
gptx repeat > README.md
```

If the above-mentioned aliases are set, you can use GPTX as follows:

```bash
gpt4 What is this website about?\n\n{{ https://ggml.ai/ }}
...
gpt4r And what is the ggml way according to the website?
```

Or (piping untrusted commands into your shell is always a great idea!):

Also cool:

```bash
gptbash Fastest way to brick my Linux machine | sh
```

## License

This project is licensed under the MIT License.
