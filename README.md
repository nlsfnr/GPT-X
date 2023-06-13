# GPTX

GPTX is a Python module that allows you to access OpenAI's Chat-models from the
terminal. It provides a command-line interface to manage conversations, and
generate responses.

## Installation

1. Clone the repository.
2. Install the required dependencies `pip install -r requirements.txt`
3. Paste your OpenAI API key into `~/.gptx/api-key.txt`

I recommend adding the following aliases to your `.bashrc` or `.bash_aliases`:

```bash
alias gpt4="gptx q --model gpt-4"  # To ask a question in a new conversation
alias gpt4r="gptx q --model gpt-4 --conversation latest"  # Continue the latest conversation
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

1. Query GPT-4:

```bash
gptx q What does this code do?\n\n{{ gptx.py }}
```

2. List conversations:

```bash
gptx ls
```

3. Remove a conversation:

```bash
gptx rm [conversation_id]
```

4. Print a conversation:

```bash
gptx print [conversation_id]
```

5. Repeat the latest message in a conversation:

```bash
gptx repeat [conversation_id]
```

If the above-mentioned aliases are set, you can use GPTX as follows:

```bash
gpt4 What is this website about?\n\n{{ https://ggml.ai/ }}
gpt4r And what is the ggml way according to the website?
```

## License

This project is licensed under the MIT License.
