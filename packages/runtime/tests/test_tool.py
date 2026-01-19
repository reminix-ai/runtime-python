"""Tests for the @tool decorator and Tool class."""

import pytest

from reminix_runtime import (
    Tool,
    ToolBase,
    ToolExecuteRequest,
    tool,
)


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_tool_decorator_creates_tool_instance(self):
        """@tool decorator returns a Tool instance."""

        @tool
        async def my_tool(param: str) -> dict:
            """A test tool."""
            return {"result": param}

        assert isinstance(my_tool, Tool)

    def test_tool_decorator_with_arguments(self):
        """@tool decorator accepts name and description arguments."""

        @tool(name="custom-name", description="Custom description")
        async def my_tool(param: str) -> dict:
            return {"result": param}

        assert my_tool.name == "custom-name"
        assert my_tool.description == "Custom description"

    def test_tool_name_defaults_to_function_name(self):
        """Tool name defaults to function name."""

        @tool
        async def get_weather(location: str) -> dict:
            """Get weather."""
            return {}

        assert get_weather.name == "get_weather"

    def test_tool_description_from_docstring(self):
        """Tool description is extracted from docstring."""

        @tool
        async def get_weather(location: str) -> dict:
            """Get the current weather for a location."""
            return {}

        assert get_weather.description == "Get the current weather for a location."

    def test_tool_description_multiline_docstring(self):
        """Tool description uses first paragraph of multiline docstring."""

        @tool
        async def get_weather(location: str) -> dict:
            """Get the current weather.

            This is additional info that should not be included.
            """
            return {}

        assert get_weather.description == "Get the current weather."


class TestToolSchemaExtraction:
    """Tests for schema extraction from function signatures."""

    def test_string_parameter(self):
        """String type hint is converted to JSON Schema string."""

        @tool
        async def my_tool(name: str) -> dict:
            """Test tool."""
            return {}

        assert my_tool.parameters.properties["name"]["type"] == "string"
        assert "name" in my_tool.parameters.required

    def test_int_parameter(self):
        """Int type hint is converted to JSON Schema integer."""

        @tool
        async def my_tool(count: int) -> dict:
            """Test tool."""
            return {}

        assert my_tool.parameters.properties["count"]["type"] == "integer"

    def test_float_parameter(self):
        """Float type hint is converted to JSON Schema number."""

        @tool
        async def my_tool(value: float) -> dict:
            """Test tool."""
            return {}

        assert my_tool.parameters.properties["value"]["type"] == "number"

    def test_bool_parameter(self):
        """Bool type hint is converted to JSON Schema boolean."""

        @tool
        async def my_tool(flag: bool) -> dict:
            """Test tool."""
            return {}

        assert my_tool.parameters.properties["flag"]["type"] == "boolean"

    def test_list_parameter(self):
        """List type hint is converted to JSON Schema array."""

        @tool
        async def my_tool(items: list) -> dict:
            """Test tool."""
            return {}

        assert my_tool.parameters.properties["items"]["type"] == "array"

    def test_dict_parameter(self):
        """Dict type hint is converted to JSON Schema object."""

        @tool
        async def my_tool(data: dict) -> dict:
            """Test tool."""
            return {}

        assert my_tool.parameters.properties["data"]["type"] == "object"

    def test_required_parameter(self):
        """Parameters without defaults are required."""

        @tool
        async def my_tool(required_param: str) -> dict:
            """Test tool."""
            return {}

        assert "required_param" in my_tool.parameters.required

    def test_optional_parameter_with_default(self):
        """Parameters with defaults are not required."""

        @tool
        async def my_tool(optional_param: str = "default") -> dict:
            """Test tool."""
            return {}

        assert "optional_param" not in my_tool.parameters.required
        assert my_tool.parameters.properties["optional_param"]["default"] == "default"

    def test_mixed_parameters(self):
        """Mix of required and optional parameters."""

        @tool
        async def my_tool(required: str, optional: str = "default") -> dict:
            """Test tool."""
            return {}

        assert "required" in my_tool.parameters.required
        assert "optional" not in my_tool.parameters.required
        assert my_tool.parameters.properties["optional"]["default"] == "default"

    def test_multiple_parameters(self):
        """Multiple parameters are all extracted."""

        @tool
        async def my_tool(name: str, age: int, active: bool = True) -> dict:
            """Test tool."""
            return {}

        assert len(my_tool.parameters.properties) == 3
        assert my_tool.parameters.properties["name"]["type"] == "string"
        assert my_tool.parameters.properties["age"]["type"] == "integer"
        assert my_tool.parameters.properties["active"]["type"] == "boolean"


class TestToolMetadata:
    """Tests for tool metadata."""

    def test_metadata_contains_type(self):
        """Metadata includes type: tool."""

        @tool
        async def my_tool(param: str) -> dict:
            """Test tool."""
            return {}

        assert my_tool.metadata["type"] == "tool"

    def test_metadata_contains_description(self):
        """Metadata includes description."""

        @tool
        async def my_tool(param: str) -> dict:
            """My tool description."""
            return {}

        assert my_tool.metadata["description"] == "My tool description."

    def test_metadata_contains_parameters(self):
        """Metadata includes parameters schema."""

        @tool
        async def my_tool(param: str) -> dict:
            """Test tool."""
            return {}

        assert "parameters" in my_tool.metadata
        assert my_tool.metadata["parameters"]["type"] == "object"

    def test_metadata_contains_output_from_return_type(self):
        """Metadata includes output schema extracted from return type hint."""

        @tool
        async def my_tool(param: str) -> dict:
            """Test tool."""
            return {"result": param}

        assert "output" in my_tool.metadata
        assert my_tool.metadata["output"]["type"] == "object"
        assert my_tool.output == {"type": "object"}

    def test_metadata_output_with_specific_types(self):
        """Output schema correctly maps Python types to JSON Schema."""

        @tool
        async def string_tool(param: str) -> str:
            """Returns a string."""
            return param

        assert string_tool.metadata["output"]["type"] == "string"

        @tool
        async def int_tool(param: str) -> int:
            """Returns an int."""
            return 42

        assert int_tool.metadata["output"]["type"] == "integer"

        @tool
        async def list_tool(param: str) -> list:
            """Returns a list."""
            return [1, 2, 3]

        assert list_tool.metadata["output"]["type"] == "array"

    def test_metadata_no_output_without_return_type(self):
        """Metadata does not include output if no return type hint."""

        @tool
        async def my_tool(param: str):
            """Test tool without return type."""
            return {"result": param}

        assert "output" not in my_tool.metadata
        assert my_tool.output is None


class TestToolExecute:
    """Tests for tool execution."""

    @pytest.mark.asyncio
    async def test_execute_async_function(self):
        """Execute calls async function with input."""

        @tool
        async def greet(name: str) -> dict:
            """Greet someone."""
            return {"message": f"Hello, {name}!"}

        request = ToolExecuteRequest(input={"name": "World"})
        response = await greet.execute(request)

        assert response.output == {"message": "Hello, World!"}
        assert response.error is None

    @pytest.mark.asyncio
    async def test_execute_sync_function(self):
        """Execute works with sync functions too."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        request = ToolExecuteRequest(input={"a": 2, "b": 3})
        response = await add.execute(request)

        assert response.output == 5
        assert response.error is None

    @pytest.mark.asyncio
    async def test_execute_with_default_values(self):
        """Execute uses default values when not provided."""

        @tool
        async def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting}, {name}!"

        request = ToolExecuteRequest(input={"name": "World"})
        response = await greet.execute(request)

        assert response.output == "Hello, World!"

    @pytest.mark.asyncio
    async def test_execute_override_default_values(self):
        """Execute can override default values."""

        @tool
        async def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting}, {name}!"

        request = ToolExecuteRequest(input={"name": "World", "greeting": "Hi"})
        response = await greet.execute(request)

        assert response.output == "Hi, World!"

    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        """Execute passes context to function if needed."""

        @tool
        async def my_tool(param: str) -> dict:
            """Test tool."""
            return {"param": param}

        request = ToolExecuteRequest(input={"param": "test"}, context={"user_id": "123"})
        response = await my_tool.execute(request)

        assert response.output == {"param": "test"}


class TestToolErrorHandling:
    """Tests for error handling in tool execution."""

    @pytest.mark.asyncio
    async def test_execute_catches_exceptions(self):
        """Execute catches exceptions and returns error."""

        @tool
        async def failing_tool(param: str) -> dict:
            """A tool that fails."""
            raise ValueError("Something went wrong")

        request = ToolExecuteRequest(input={"param": "test"})
        response = await failing_tool.execute(request)

        assert response.output is None
        assert response.error == "Something went wrong"

    @pytest.mark.asyncio
    async def test_execute_missing_required_parameter(self):
        """Execute returns error when required parameter is missing."""

        @tool
        async def my_tool(required_param: str) -> dict:
            """Test tool."""
            return {"result": required_param}

        request = ToolExecuteRequest(input={})
        response = await my_tool.execute(request)

        assert response.output is None
        assert response.error is not None
        assert "required_param" in response.error


class TestToolBase:
    """Tests for ToolBase abstract class."""

    def test_tool_inherits_from_toolbase(self):
        """Tool class inherits from ToolBase."""

        @tool
        async def my_tool(param: str) -> dict:
            """Test tool."""
            return {}

        assert isinstance(my_tool, ToolBase)

    def test_toolbase_is_abstract(self):
        """ToolBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ToolBase()
