from typing import Any, Callable, Union, get_origin, get_args
from inspect import isclass
from livekit_mcp_client.utils import sanitize_name,translate_type
import typing 
class TypeTranslator:
    """Handles complex JSON Schema type conversions to Python types"""
    def __init__(self, custom_translator: Callable[[str], str] = None):
        self._base_translator = custom_translator or self._default_translator
    
    def _default_translator(self, json_type: str) -> str:
        type_map = {
            'string': 'str',
            'integer': 'int',
            'number': 'float',
            'boolean': 'bool',
            'object': 'dict',
            'array' : 'list',
            'null': 'None'
        }
        if json_type is None:
            return 'Any'
        return type_map.get(json_type, 'Any')
    
    def translate(self, schema: dict) -> str:
        """Convert JSON Schema type definition to Python type string"""
        json_type = schema.get('type', 'Any')
        
        if isinstance(json_type, list):
            py_types = [self._translate_single_type(t, schema) for t in json_type]
            return f"Union[{', '.join(py_types)}]"
        
        return self._translate_single_type(json_type, schema)
    
    def _translate_single_type(self, json_type: str, schema: dict) -> str:
        if json_type == 'array':
            items_schema = schema.get('items', {})
            item_type = self.translate(items_schema)
            return f"list[{item_type}]"
        return self._base_translator(json_type)

class SchemaReader:
    """Reads and processes JSON Schema parameters"""
    
    def __init__(self, translator: TypeTranslator = None):
        self.translator = translator or TypeTranslator()
    
    def get_parameters(self, input_schema: dict) -> tuple[list, list, dict]:
        """Returns (required_params, optional_params, param_map)"""
        if not isinstance(input_schema.get("properties"), dict):
          raise ValueError("Schema must have 'properties' as a dict")
        props = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))
        
        required_params = []
        optional_params = []
        param_map = {}
        
        for orig_name, prop in props.items():
            sanitized = sanitize_name(orig_name)
            if sanitized in param_map:
              sanitized = f"{sanitized}_{hash(orig_name)}"
            param_map[sanitized] = orig_name
            
            py_type = self.translator.translate(prop)
            param_def = f"{sanitized}: {py_type}"
            
            if orig_name not in required:
                param_def += " = None"
            
            (required_params if orig_name in required else optional_params).append(param_def)
        
        return required_params, optional_params, param_map
    
