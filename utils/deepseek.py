import re


def format_markdown_blockquote(text: str) -> str:
    """Formats a given text as a Markdown blockquote."""
    return "\n".join(f"> {line}" for line in text.split("\n"))


def format_deepseek_message(bot_message):
    """Formats DeepSeek R1 output to Markdown blockquotes for <think> sections."""
    # bot_message = bot_message.replace('<think>', '```markdown').replace('</think>', '```')
    # Replace <think>...</think> with blockquote format
    bot_message = re.sub(
        r"<think>(.*?)</think>",
        lambda m: format_markdown_blockquote(m.group(1).strip()),
        bot_message,
        flags=re.DOTALL,
    )
    # Handle open <think> tags (incomplete)
    bot_message = re.sub(
        r"<think>(.*)",
        lambda m: format_markdown_blockquote(m.group(1).strip()),
        bot_message,
        flags=re.DOTALL,
    )
    return bot_message
