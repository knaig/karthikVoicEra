"""Tool calling — @call.tool decorator and OpenAI function-calling integration."""

import inspect
import json
from typing import Callable, Any


# Python type -> JSON Schema type
_TYPE_MAP = {
    str: "string", int: "integer", float: "number", bool: "boolean",
}


class ToolRegistry:
    """Stores tool definitions and handlers for OpenAI function calling."""

    def __init__(self):
        self._tools: list[dict] = []
        self._handlers: dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        """Register a function as a callable tool.

        Inspects the function signature and docstring to generate
        an OpenAI-compatible function-calling tool schema.
        """
        sig = inspect.signature(func)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
            json_type = _TYPE_MAP.get(param_type, "string")
            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        tool_schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": (func.__doc__ or "").strip() or f"Call {name}",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

        self._tools.append(tool_schema)
        self._handlers[name] = func

    def get_openai_tools(self) -> list[dict]:
        return self._tools

    def get_handler(self, name: str) -> Callable | None:
        return self._handlers.get(name)

    def has_tools(self) -> bool:
        return len(self._tools) > 0

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool handler and return the result as a string."""
        handler = self._handlers.get(name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            result = handler(**arguments)
            # Support async handlers
            if inspect.isawaitable(result):
                result = await result
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})
