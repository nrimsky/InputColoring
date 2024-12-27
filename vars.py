assistant_fake_tags = [
    "assistant",
    "ai",
    "a",
    "AI",
    "A",
    "Assistant",
    "AIAssistant",
]

user_fake_tags = [
    "user",
    "human",
    "u",
    "h",
    "Human",
    "User",
    "user_input",
    "human_input",
    "U"
]

format_functions = [
    lambda x: x + ":",
    lambda x: x + ">",
    lambda x: f"<{x}>",
    lambda x: f"({x})",
    lambda x: f"[{x}]",
    lambda x: f"<|{x}|>",
    lambda x: f"({x}):",
    lambda x: f"{x} >",
    lambda x: f"{x} >>",
    lambda x: f"|{x}|",
]
