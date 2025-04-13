import pytest
from typing import Any, Union
from livekit_mcp_client.clients.schema import TypeTranslator, SchemaReader
from livekit_mcp_client import utils


@pytest.fixture
def translator() -> TypeTranslator:
    return TypeTranslator()

@pytest.mark.parametrize("json_type, expected_py_type", [
    ('string', 'str'),
    ('integer', 'int'),
    ('number', 'float'),
    ('boolean', 'bool'),
    ('object', 'dict'),
    ('null', 'None'),
    ('unknown', 'Any'),
])
def test_translator_basic_types(translator: TypeTranslator, json_type: str, expected_py_type: str):
    schema = {'type': json_type}
    assert translator.translate(schema) == expected_py_type

def test_translator_array_type(translator: TypeTranslator):
    schema = {'type': 'array', 'items': {'type': 'string'}}
    assert translator.translate(schema) == 'list[str]'

def test_translator_array_no_items(translator: TypeTranslator):
    schema = {'type': 'array'}
    assert translator.translate(schema) == 'list[Any]'

def test_translator_array_complex_items(translator: TypeTranslator):
    schema = {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'integer'}}}
    assert translator.translate(schema) == 'list[list[int]]'

def test_translator_union_type(translator: TypeTranslator):
    schema = {'type': ['string', 'null']}
    # Order might vary, check both possibilities
    expected1 = 'Union[str, None]'
    expected2 = 'Union[None, str]'
    result = translator.translate(schema)
    assert result == expected1 or result == expected2

def test_translator_union_complex_types(translator: TypeTranslator):
     schema = {'type': ['integer', 'string', 'null']}
     result = translator.translate(schema)
     # Use sets for order-independent comparison
     expected_types = {'int', 'str', 'None'}
     import typing
     # Parse the generated Union string
     parsed_union = eval(result, {"Union": typing.Union, "str":str, "int":int, "None":type(None)})
     assert typing.get_origin(parsed_union) is typing.Union
     assert set(typing.get_args(parsed_union)) == {int, str, type(None)}


def test_translator_default_type(translator: TypeTranslator):
    schema = {}
    assert translator.translate(schema) == 'Any'

def test_translator_custom_translator():
    def custom(json_type: str) -> str:
        return "custom_" + json_type
    custom_translator = TypeTranslator(custom_translator=custom)
    schema = {'type': 'string'}
    assert custom_translator.translate(schema) == 'custom_string'
    schema_array = {'type': 'array', 'items': {'type': 'integer'}}
    assert custom_translator.translate(schema_array) == 'list[custom_integer]'


# --- Test SchemaReader ---

@pytest.fixture
def reader() -> SchemaReader:
    return SchemaReader()

def test_reader_basic_schema(reader: SchemaReader):
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"}
        },
        "required": ["name", "age"]
    }
    req, opt, pmap = reader.get_parameters(schema)

    assert sorted(req) == sorted(["name: str", "age: int"])
    assert opt == ["city: str = None"]
    assert pmap == {"name": "name", "age": "age", "city": "city"}

def test_reader_no_required(reader: SchemaReader):
    schema = {
        "type": "object",
        "properties": {
            "optional_param": {"type": "boolean"}
        }
        # "required" is omitted
    }
    req, opt, pmap = reader.get_parameters(schema)
    assert req == []
    assert opt == ["optional_param: bool = None"]
    assert pmap == {"optional_param": "optional_param"}

def test_reader_all_required(reader: SchemaReader):
    schema = {
        "type": "object",
        "properties": {
            "param1": {"type": "number"}
        },
        "required": ["param1"]
    }
    req, opt, pmap = reader.get_parameters(schema)
    assert req == ["param1: float"]
    assert opt == []
    assert pmap == {"param1": "param1"}

def test_reader_complex_types(reader: SchemaReader):
    schema = {
        "type": "object",
        "properties": {
            "user_ids": {"type": "array", "items": {"type": "integer"}},
            "maybe_name": {"type": ["string", "null"]},
        },
        "required": ["user_ids"]
    }
    req, opt, pmap = reader.get_parameters(schema)

    assert req == ["user_ids: list[int]"]
    assert len(opt) == 1
    assert opt[0].startswith("maybe_name: Union[") and opt[0].endswith("] = None")
    assert "str" in opt[0] and "None" in opt[0]

    assert pmap == {"user_ids": "user_ids", "maybe_name": "maybe_name"}

def test_reader_sanitization(reader: SchemaReader, monkeypatch):
    def mock_sanitize(name):
        return name.replace("-", "_").replace(" ", "_")
    monkeypatch.setattr(utils, "sanitize_name", mock_sanitize)

    schema = {
        "type": "object",
        "properties": {
            "first-name": {"type": "string"},
            "last name": {"type": "string"}
        },
        "required": ["first-name"]
    }
    req, opt, pmap = reader.get_parameters(schema)

    assert req == ["first_name: str"]
    assert opt == ["last_name: str = None"]
    assert pmap == {"first_name": "first-name", "last_name": "last name"}

def test_reader_sanitization_collision(reader: SchemaReader, monkeypatch):
    def mock_sanitize_collision(name):
        if name == "param-a" or name == "param_a":
            return "param_a"
        return name
    monkeypatch.setattr(utils, "sanitize_name", mock_sanitize_collision)

    schema = {
        "type": "object",
        "properties": {
            "param-a": {"type": "string"},
            "param_a": {"type": "integer"}
        },
        "required": ["param-a"]
    }
    req, opt, pmap = reader.get_parameters(schema)

    assert len(req) == 1
    assert len(opt) == 1
    assert req[0].startswith("param_a:")

    assert any(p.startswith("param_a_") and p.endswith(": int = None") for p in opt)

    assert len(pmap) == 2
    assert "param_a" in pmap
    assert pmap["param_a"] == "param-a"
    assert any(k.startswith("param_a_") and v == "param_a" for k, v in pmap.items())


def test_reader_empty_properties(reader: SchemaReader):
    schema = {
        "type": "object",
        "properties": {}
    }
    req, opt, pmap = reader.get_parameters(schema)
    assert req == []
    assert opt == []
    assert pmap == {}

def test_reader_invalid_schema_no_properties(reader: SchemaReader):
     schema = {"type": "object"}
     with pytest.raises(ValueError, match="Schema must have 'properties' as a dict"):
         reader.get_parameters(schema)

def test_reader_invalid_schema_properties_not_dict(reader: SchemaReader):
     schema = {"type": "object", "properties": ["invalid"]}
     with pytest.raises(ValueError, match="Schema must have 'properties' as a dict"):
         reader.get_parameters(schema)