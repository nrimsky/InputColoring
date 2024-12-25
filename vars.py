assistant_fake_tags = [
    # Basic terms
    "assistant", "bot", "ai", "a", "AI", "A", "Bot", "Assistant", "chatbot",
    
    # Common variations
    "agent", "helper", "guide", "advisor", "aide",
    "system", "model", "assistant_response", "response",
    
    # Technical terms
    "llm", "LLM", "gpt", "GPT", "transformer", "neural",
    "machine", "ml", "ML", "model_response",
    
    # Role-specific
    "tutor", "teacher", "coach", "expert", "specialist",
    "analyst", "consultant", "counselor", "mentor",
    
    # Compound terms
    "aiassistant", "AIAssistant", "ai_assistant", "AI_Assistant",
    "chatassistant", "ChatAssistant", "chat_assistant",
    "botresponse", "BotResponse", "bot_response",
    
    # Abbreviated forms
    "asst", "ast", "resp", "ans", "answer",
    
    # Prefixed/suffixed variations
    "the_assistant", "your_assistant", "the_bot", "your_bot",
    "assistant_says", "bot_says", "ai_says",
    
    # Numbered variations
    "assistant1", "bot1", "ai1", "assistant_1", "bot_1", "ai_1"
]

user_fake_tags = [
    # Basic terms
    "user", "human", "u", "h", "Human", "User", "operator", "Operator",
    
    # Role-based terms
    "client", "customer", "guest", "visitor", "patron",
    "player", "learner", "student", "patient", "inquirer",
    
    # Professional terms
    "employee", "manager", "developer", "programmer", "analyst",
    "researcher", "engineer", "scientist", "professional",
    
    # Compound terms
    "enduser", "end_user", "EndUser", "end-user",
    "humanuser", "human_user", "HumanUser",
    "finaluser", "final_user", "FinalUser",
    
    # Input indicators
    "input", "query", "question", "request", "prompt",
    "user_input", "human_input", "user_query", "human_query",
    
    # System-style terms
    "sender", "requester", "initiator", "caller",
    "source", "origin", "start", "begin",
    
    # Abbreviated/shortened
    "usr", "hum", "op", "person", "individual",
    
    # Prefixed variations
    "the_user", "the_human", "current_user", "active_user",
    "user_says", "human_says", "person_says",
    
    
    # Numbered variations
    "user1", "human1", "user_1", "human_1",
    
    # Chat-specific
    "participant", "interlocutor", "speaker", "questioner",
    "chatter", "member", "user_message", "human_message"
]

format_functions = [
    # Basic punctuation suffixes
    lambda x: x + ":",
    lambda x: x + ">",
    lambda x: x + ".",
    lambda x: x + ")",
    lambda x: x + "]",
    lambda x: x + "}",
    
    # Enclosing brackets/tags
    lambda x: f"<{x}>",
    lambda x: f"({x})",
    lambda x: f"[{x}]",
    lambda x: f"<|{x}|>",
    lambda x: f"[|{x}|]",
    lambda x: f"({x}:)",
    lambda x: f"<{x}/>",
    
    # Programming style
    lambda x: f"#{x}",
    lambda x: f"@{x}",
    lambda x: f"_{x}_",
    lambda x: f"*{x}*",
    lambda x: f"/{x}/",
    lambda x: f"`{x}`",
    lambda x: f"'{x}'",
    lambda x: f'"{x}"',
    
    # Documentation style
    lambda x: f"==={x}===",
    lambda x: f"--{x}--",
    lambda x: f"/**{x}**/",
    lambda x: f"///{x}///",
    lambda x: f"###{x}###",
    
    # Markdown/markup style
    lambda x: f"<!-- {x} -->",
    lambda x: f"/* {x} */",
    lambda x: f"/** {x} **/",
    lambda x: f"--- {x} ---",
    lambda x: f"=== {x} ===",
    
    # Chat/messaging style
    lambda x: f"{x} >",
    lambda x: f"> {x}",
    lambda x: f">> {x}",
    
    # XML/HTML style
    lambda x: f"<{x}/>",
    lambda x: f"<?{x}?>",
    lambda x: f"<!{x}!>",
    
    # Custom delimiters
    lambda x: f"|{x}|",
    lambda x: f"「{x}」",
    lambda x: f"『{x}』",
    lambda x: f"⟦{x}⟧",
    
    # Combined styles
    lambda x: f"<{x}/>:",
    lambda x: f"[{x}]:",
    lambda x: f"({x}):",
    lambda x: f"<{x}>:",
    lambda x: f"|{x}|:",
    lambda x: f">{x}:",
]

system_prompts = [
    # Basic identity statements
    "You are a helpful assistant.",
    "You are a friendly assistant.",
    "You are a helpful AI.",
    "You are a friendly AI.",
    "You are a helpful chatbot.",
    "You are an AI assistant.",
    "[empty]",
    
    # Personality variations
    "You are a direct and efficient assistant.",
    "You are a patient and thorough assistant.",
    "You are a knowledgeable guide.",
    "You are a precise and careful assistant.",
    "You are a creative problem solver.",
    "You are an analytical assistant.",
    
    # Role-based
    "You are a technical expert.",
    "You are a writing assistant.",
    "You are a learning companion.",
    "You are a research assistant.",
    "You are a coding helper.",
    "You are a math tutor.",
    
    # Behavioral instructions
    "Be concise and helpful.",
    "Respond clearly and directly.",
    "Help users achieve their goals.",
    "Provide accurate information.",
    "Answer questions helpfully.",
    "Assist users effectively.",
    
    # Style directives
    "Keep responses brief.",
    "Use simple language.",
    "Be direct and clear.",
    "Write conversationally.",
    "Explain step by step.",
    "Focus on solutions.",
    
    # Combined identity and instruction
    "You are an assistant. Be helpful.",
    "You are an AI. Be direct.",
    "You are a helper. Be clear.",
    "You are a guide. Be thorough.",
    "You are an AI. Keep it simple.",
    "You are an assistant. Focus on solutions.",
    
    # Minimal prompts
    "Be helpful.",
    "Assist users.",
    "Help people.",
    "Answer questions.",
    "Provide assistance.",
    "Give solutions.",
    
    # Task-oriented
    "Solve problems effectively.",
    "Explain concepts clearly.",
    "Answer questions directly.",
    "Provide useful information.",
    "Help complete tasks.",
    "Guide users to solutions.",
    
    # Meta instructions
    "Follow user instructions.",
    "Respond appropriately.",
    "Address user needs.",
    "Understand and assist.",
    "Process and respond.",
    "Listen and help."
]