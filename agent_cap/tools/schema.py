# Tool schema for LLM tool-calling (OpenAI-compatible)

calculator_tool_schema = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Safely evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expr"]
            }
        }
    }
]
