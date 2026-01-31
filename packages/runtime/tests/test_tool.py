"""Tests for the @tool decorator and Tool class."""

from typing import TypedDict

import pytest
from pydantic import BaseModel, Field

from reminix_runtime import (
    Tool,
    ToolBase,
    ToolCallRequest,
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

        assert my_tool.input["properties"]["name"]["type"] == "string"
        assert "name" in my_tool.input["required"]

    def test_int_parameter(self):
        """Int type hint is converted to JSON Schema integer."""

        @tool
        async def my_tool(count: int) -> dict:
            """Test tool."""
            return {}

        assert my_tool.input["properties"]["count"]["type"] == "integer"

    def test_float_parameter(self):
        """Float type hint is converted to JSON Schema number."""

        @tool
        async def my_tool(value: float) -> dict:
            """Test tool."""
            return {}

        assert my_tool.input["properties"]["value"]["type"] == "number"

    def test_bool_parameter(self):
        """Bool type hint is converted to JSON Schema boolean."""

        @tool
        async def my_tool(flag: bool) -> dict:
            """Test tool."""
            return {}

        assert my_tool.input["properties"]["flag"]["type"] == "boolean"

    def test_list_parameter(self):
        """List type hint is converted to JSON Schema array."""

        @tool
        async def my_tool(items: list) -> dict:
            """Test tool."""
            return {}

        assert my_tool.input["properties"]["items"]["type"] == "array"

    def test_dict_parameter(self):
        """Dict type hint is converted to JSON Schema object."""

        @tool
        async def my_tool(data: dict) -> dict:
            """Test tool."""
            return {}

        assert my_tool.input["properties"]["data"]["type"] == "object"

    def test_required_parameter(self):
        """Parameters without defaults are required."""

        @tool
        async def my_tool(required_param: str) -> dict:
            """Test tool."""
            return {}

        assert "required_param" in my_tool.input["required"]

    def test_optional_parameter_with_default(self):
        """Parameters with defaults are not required."""

        @tool
        async def my_tool(optional_param: str = "default") -> dict:
            """Test tool."""
            return {}

        assert "optional_param" not in my_tool.input["required"]
        assert my_tool.input["properties"]["optional_param"]["default"] == "default"

    def test_mixed_parameters(self):
        """Mix of required and optional parameters."""

        @tool
        async def my_tool(required: str, optional: str = "default") -> dict:
            """Test tool."""
            return {}

        assert "required" in my_tool.input["required"]
        assert "optional" not in my_tool.input["required"]
        assert my_tool.input["properties"]["optional"]["default"] == "default"

    def test_multiple_parameters(self):
        """Multiple parameters are all extracted."""

        @tool
        async def my_tool(name: str, age: int, active: bool = True) -> dict:
            """Test tool."""
            return {}

        assert len(my_tool.input["properties"]) == 3
        assert my_tool.input["properties"]["name"]["type"] == "string"
        assert my_tool.input["properties"]["age"]["type"] == "integer"
        assert my_tool.input["properties"]["active"]["type"] == "boolean"


class TestToolMetadata:
    """Tests for tool metadata."""

    def test_metadata_contains_description(self):
        """Metadata includes description."""

        @tool
        async def my_tool(param: str) -> dict:
            """My tool description."""
            return {}

        assert my_tool.metadata["description"] == "My tool description."

    def test_metadata_contains_input(self):
        """Metadata includes input schema."""

        @tool
        async def my_tool(param: str) -> dict:
            """Test tool."""
            return {}

        assert "input" in my_tool.metadata
        assert my_tool.metadata["input"]["type"] == "object"

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

    def test_metadata_default_output(self):
        """Metadata has default output schema."""

        @tool
        async def my_tool(param: str):
            """Test tool without return type."""
            return {"result": param}

        assert my_tool.output == {"type": "string"}


class TestPydanticOutputSchema:
    """Tests for Pydantic model output schema extraction."""

    def test_pydantic_model_output_schema(self):
        """Pydantic model return type generates full JSON schema."""

        class GreetOutput(BaseModel):
            message: str

        @tool
        def greet(name: str) -> GreetOutput:
            """Greet someone."""
            return GreetOutput(message=f"Hello, {name}!")

        output = greet.output
        assert output is not None
        assert output["type"] == "object"
        assert "properties" in output
        assert "message" in output["properties"]
        assert output["properties"]["message"]["type"] == "string"
        assert output["required"] == ["message"]

    def test_pydantic_model_with_field_descriptions(self):
        """Pydantic Field descriptions are included in schema."""

        class WeatherOutput(BaseModel):
            temp: int = Field(description="Temperature in degrees")
            condition: str = Field(description="Weather condition")

        @tool
        def get_weather(location: str) -> WeatherOutput:
            """Get weather."""
            return WeatherOutput(temp=72, condition="sunny")

        output = get_weather.output
        assert output["properties"]["temp"]["description"] == "Temperature in degrees"
        assert output["properties"]["condition"]["description"] == "Weather condition"

    def test_pydantic_model_with_optional_fields(self):
        """Pydantic models with optional fields have correct required list."""

        class OutputWithOptional(BaseModel):
            required_field: str
            optional_field: str = "default"

        @tool
        def my_tool(param: str) -> OutputWithOptional:
            """Test tool."""
            return OutputWithOptional(required_field=param)

        output = my_tool.output
        assert "required_field" in output["required"]
        assert "optional_field" not in output["required"]

    @pytest.mark.asyncio
    async def test_pydantic_model_execution(self):
        """Tool with Pydantic return type executes correctly."""

        class GreetOutput(BaseModel):
            message: str

        @tool
        def greet(name: str) -> GreetOutput:
            """Greet someone."""
            return GreetOutput(message=f"Hello, {name}!")

        request = ToolCallRequest(input={"name": "World"})
        response = await greet.call(request)

        assert response.output.message == "Hello, World!"


class TestTypedDictOutputSchema:
    """Tests for TypedDict output schema extraction."""

    def test_typeddict_output_schema(self):
        """TypedDict return type generates JSON schema with properties."""

        class GreetOutput(TypedDict):
            message: str

        @tool
        def greet(name: str) -> GreetOutput:
            """Greet someone."""
            return {"message": f"Hello, {name}!"}

        output = greet.output
        assert output is not None
        assert output["type"] == "object"
        assert "properties" in output
        assert "message" in output["properties"]
        assert output["properties"]["message"]["type"] == "string"

    def test_typeddict_with_multiple_fields(self):
        """TypedDict with multiple fields extracts all properties."""

        class WeatherOutput(TypedDict):
            temp: int
            condition: str
            location: str

        @tool
        def get_weather(location: str) -> WeatherOutput:
            """Get weather."""
            return {"temp": 72, "condition": "sunny", "location": location}

        output = get_weather.output
        assert len(output["properties"]) == 3
        assert output["properties"]["temp"]["type"] == "integer"
        assert output["properties"]["condition"]["type"] == "string"
        assert output["properties"]["location"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_typeddict_execution(self):
        """Tool with TypedDict return type executes correctly."""

        class GreetOutput(TypedDict):
            message: str

        @tool
        def greet(name: str) -> GreetOutput:
            """Greet someone."""
            return {"message": f"Hello, {name}!"}

        request = ToolCallRequest(input={"name": "World"})
        response = await greet.call(request)

        assert response.output == {"message": "Hello, World!"}


class TestDocstringParsing:
    """Tests for docstring parsing for parameter descriptions."""

    def test_google_style_docstring_parameter_descriptions(self):
        """Parameter descriptions are extracted from Google-style docstrings."""

        @tool
        def greet(name: str, greeting: str = "Hello") -> str:
            """Generate a greeting.

            Args:
                name: The name of the person to greet
                greeting: The greeting to use
            """
            return f"{greeting}, {name}!"

        props = greet.input["properties"]
        assert props["name"]["description"] == "The name of the person to greet"
        assert props["greeting"]["description"] == "The greeting to use"

    def test_numpy_style_docstring_parameter_descriptions(self):
        """Parameter descriptions are extracted from NumPy-style docstrings."""

        @tool
        def calculate(a: float, b: float) -> float:
            """Perform calculation.

            Parameters
            ----------
            a : float
                First operand
            b : float
                Second operand
            """
            return a + b

        props = calculate.input["properties"]
        assert props["a"]["description"] == "First operand"
        assert props["b"]["description"] == "Second operand"

    def test_return_description_added_to_output_schema(self):
        """Return description from docstring is added to output schema."""

        @tool
        def greet(name: str) -> str:
            """Generate a greeting.

            Args:
                name: The person's name

            Returns:
                A personalized greeting message
            """
            return f"Hello, {name}!"

        assert greet.output is not None
        assert greet.output["description"] == "A personalized greeting message"

    def test_docstring_without_args_section(self):
        """Tools work fine without Args section in docstring."""

        @tool
        def simple_tool(param: str) -> str:
            """A simple tool."""
            return param

        # Should not have description but should work
        assert "description" not in simple_tool.input["properties"]["param"]

    def test_partial_docstring_args(self):
        """Only documented parameters get descriptions."""

        @tool
        def partial_docs(a: str, b: str, c: str) -> str:
            """Tool with partial docs.

            Args:
                a: First parameter
                c: Third parameter
            """
            return a + b + c

        props = partial_docs.input["properties"]
        assert props["a"]["description"] == "First parameter"
        assert "description" not in props["b"]
        assert props["c"]["description"] == "Third parameter"


class TestToolExecute:
    """Tests for tool execution."""

    @pytest.mark.asyncio
    async def test_execute_async_function(self):
        """Execute calls async function with input."""

        @tool
        async def greet(name: str) -> dict:
            """Greet someone."""
            return {"message": f"Hello, {name}!"}

        request = ToolCallRequest(input={"name": "World"})
        response = await greet.call(request)

        assert response.output == {"message": "Hello, World!"}

    @pytest.mark.asyncio
    async def test_execute_sync_function(self):
        """Execute works with sync functions too."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        request = ToolCallRequest(input={"a": 2, "b": 3})
        response = await add.call(request)

        assert response.output == 5

    @pytest.mark.asyncio
    async def test_execute_with_default_values(self):
        """Execute uses default values when not provided."""

        @tool
        async def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting}, {name}!"

        request = ToolCallRequest(input={"name": "World"})
        response = await greet.call(request)

        assert response.output == "Hello, World!"

    @pytest.mark.asyncio
    async def test_execute_override_default_values(self):
        """Execute can override default values."""

        @tool
        async def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting}, {name}!"

        request = ToolCallRequest(input={"name": "World", "greeting": "Hi"})
        response = await greet.call(request)

        assert response.output == "Hi, World!"

    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        """Execute passes context to function if needed."""

        @tool
        async def my_tool(param: str) -> dict:
            """Test tool."""
            return {"param": param}

        request = ToolCallRequest(input={"param": "test"}, context={"user_id": "123"})
        response = await my_tool.call(request)

        assert response.output == {"param": "test"}


class TestToolErrorHandling:
    """Tests for error handling in tool execution."""

    @pytest.mark.asyncio
    async def test_execute_propagates_exceptions(self):
        """Execute propagates exceptions to caller (server handles them)."""

        @tool
        async def failing_tool(param: str) -> dict:
            """A tool that fails."""
            raise ValueError("Something went wrong")

        request = ToolCallRequest(input={"param": "test"})
        with pytest.raises(ValueError, match="Something went wrong"):
            await failing_tool.call(request)

    @pytest.mark.asyncio
    async def test_execute_propagates_missing_parameter_error(self):
        """Execute propagates TypeError when required parameter is missing."""

        @tool
        async def my_tool(required_param: str) -> dict:
            """Test tool."""
            return {"result": required_param}

        request = ToolCallRequest(input={})
        with pytest.raises(TypeError, match="required_param"):
            await my_tool.call(request)


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
