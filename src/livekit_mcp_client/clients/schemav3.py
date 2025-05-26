import logging
from typing import Optional, List, Dict, Tuple, Type, Any, Union, ForwardRef
import datetime
import uuid

from livekit_mcp_client.exceptions import (
    InvalidSchemaStructureError,
    InvalidSchemaTypeError,
    SchemaRefResolutionError,
    SchemaUnsupportedTypeError,
    SchemaError,
)

from pydantic import (
    BaseModel as PydanticBaseModel,
    Field as PydanticField,
    create_model,
)

from livekit_mcp_client.utils import sanitize_name

logger = logging.getLogger(__name__)

PydanticFieldInfo = Tuple[Type, PydanticField]


class SchemaReaderV3:
    """
    Reads JSON Schemas and dynamically creates Pydantic model *objects*
    suitable for use as tool inputs, without using exec().

    This version focuses on generating Pydantic model classes directly using
    `pydantic.create_model`. It handles nested objects, arrays, basic type
    mapping, optionality, defaults, and $ref resolution within the provided
    definitions.
    """

    def __init__(self):
        """Initializes the SchemaReaderV3."""
        self._created_models_cache: Dict[str, Type[PydanticBaseModel]] = {}
        self._processing_stack: List[str] = []

    def _clear_state(self):
        """Resets the internal model cache and processing stack.
        Useful if processing multiple independent sets of schemas with the same reader instance.
        """
        self._created_models_cache.clear()
        self._processing_stack.clear()

    def _map_json_type_to_python(
        self, schema_type: str, schema_format: Optional[str] = None
    ) -> Type:
        """Maps basic JSON schema types and common formats to Python type objects.

        Args:
            schema_type: The JSON 'type' string (e.g., "string", "integer").
            schema_format: The JSON 'format' string (e.g., "date-time", "uuid").

        Returns:
            The corresponding Python type object (e.g., str, int, datetime.datetime).
            Defaults to `typing.Any` for unmapped types.

        Raises:
            SchemaUnsupportedTypeError: If the type is explicitly unsupported.
        """
        # Basic type mapping
        if schema_type == "string":
            if schema_format == "date-time":
                return datetime.datetime
            elif schema_format == "date":
                return datetime.date
            elif schema_format == "time":
                return datetime.time
            elif schema_format == "uuid":
                return uuid.UUID
            return str
        elif schema_type == "integer":
            return int
        elif schema_type == "number":
            return float  # might be decimal sometimes (keep in mind)
        elif schema_type == "boolean":
            return bool
        elif schema_type == "null":
            return type(None)

        logger.warning(
            f"Unmapped JSON schema type: {schema_type}, format: {schema_format}. Defaulting to Any."
        )
        return Any  # Default fallback

    def _schema_to_pydantic_field(
        self,
        prop_schema: Dict[str, Any],
        prop_name_hint: str,
        definitions: Dict[str, Any],
        base_model: Type[PydanticBaseModel],
        is_required: bool,
    ) -> PydanticFieldInfo:
        """
        Recursively processes a JSON schema fragment for a single property.

        Determines the appropriate Python type hint and constructs a configured
        `pydantic.Field` object encapsulating defaults, optionality, descriptions,
        and aliases based on the schema definition.

        Args:
            prop_schema: The JSON schema dictionary for the specific property.
            prop_name_hint: A descriptive name hint (e.g., "UserModel_user_id") used
                            for generating unique names for nested models.
            definitions: A dictionary containing global schema definitions, used for resolving `$ref`.
            base_model: The base Pydantic model class new models should inherit from.
            is_required: Boolean indicating if this property is listed in the parent schema's 'required' array.

        Returns:
            A tuple containing: `(resolved_python_type, configured_pydantic_field)`.

        Raises:
            InvalidSchemaTypeError: If `prop_schema` is not a dictionary.
            SchemaRefResolutionError: If a `$ref` cannot be resolved within `definitions`.
            SchemaUnsupportedTypeError: If an unsupported JSON type/format is encountered.
            InvalidSchemaStructureError: If schema structure is invalid (e.g., array missing 'items').
            SchemaCircularReferenceError: If a direct circular reference is detected during recursive calls
                                          (and ForwardRef is not being used or fails).
            Any exception raised during recursive calls to `_create_model_from_schema`.
        """
        field_type: Type = Any
        field_default: Any = ...

        if not isinstance(prop_schema, dict):
            raise InvalidSchemaTypeError(
                expected_type="dict",
                actual_type=prop_schema,
                context=f"Property schema for '{prop_name_hint}'",
            )

        if "$ref" in prop_schema:
            ref_path = prop_schema["$ref"]
            ref_name = ref_path.split("/")[-1]

            if ref_name in self._created_models_cache:
                field_type = self._created_models_cache[ref_name]
                logger.debug(
                    f"Using cached model/ref for '$ref': '{ref_path}' -> '{ref_name}'"
                )

            elif ref_name in definitions:
                logger.debug(
                    f"Resolving $ref '{ref_path}' by creating/getting model '{ref_name}'"
                )
                field_type = self._create_model_from_schema(
                    model_name=ref_name,
                    schema=definitions[ref_name],
                    base_model=base_model,
                    definitions=definitions,
                )
            else:
                raise SchemaRefResolutionError(
                    ref_path=ref_path, context=f"Property '{prop_name_hint}'"
                )

        else:
            json_type = prop_schema.get("type")
            json_format = prop_schema.get("format")

            if isinstance(json_type, list):
                py_types = []
                for t in json_type:
                    if t == "null":
                        continue
                    sub_schema = prop_schema.copy()
                    sub_schema["type"] = t

                    sub_type, _ = self._schema_to_pydantic_field(
                        sub_schema,
                        f"{prop_name_hint}_{t}",
                        definitions,
                        base_model,
                        True,
                    )
                    py_types.append(sub_type)

                unique_py_types = tuple(sorted(list(set(py_types)), key=str))
                if not unique_py_types:
                    field_type = type(None)
                elif len(unique_py_types) == 1:
                    field_type = unique_py_types[0]
                else:
                    field_type = Union[unique_py_types]

            elif json_type == "object":
                if "properties" in prop_schema:
                    nested_model_name = self._generate_unique_model_name(prop_name_hint)
                    logger.debug(
                        f"Creating nested model '{nested_model_name}' for property '{prop_name_hint}'"
                    )

                    field_type = self._create_model_from_schema(
                        model_name=nested_model_name,
                        schema=prop_schema,
                        base_model=base_model,
                        definitions=definitions,
                    )
                else:
                    logger.debug(
                        f"Object property '{prop_name_hint}' has no 'properties' defined. Defaulting to Dict[str, Any]."
                    )
                    field_type = Dict[str, Any]

            elif json_type == "array":
                items_schema = prop_schema.get("items")
                item_type: Type = Any

                if items_schema is None:
                    raise InvalidSchemaStructureError(
                        structure_element="items",
                        expected_type="dict/object",
                        actual_type=None,
                        context=f"Array property '{prop_name_hint}'",
                    )
                elif isinstance(items_schema, dict):
                    item_type, _ = self._schema_to_pydantic_field(
                        items_schema,
                        f"{prop_name_hint}_item",
                        definitions,
                        base_model,
                        True,
                    )
                else:
                    raise InvalidSchemaTypeError(
                        expected_type="dict",
                        actual_type=items_schema,
                        context=f"Array 'items' definition for '{prop_name_hint}'",
                    )

                field_type = List[item_type]

            elif isinstance(json_type, str):
                field_type = self._map_json_type_to_python(json_type, json_format)

            else:
                raise SchemaUnsupportedTypeError(
                    json_type=json_type,
                    json_format=json_format,
                    context=f"Property '{prop_name_hint}'",
                )

        if not is_required:
            field_default = prop_schema.get("default", None)

            is_already_optional = (
                field_type is Any
                or (
                    hasattr(field_type, "__origin__")
                    and field_type.__origin__ is Union
                    and type(None) in getattr(field_type, "__args__", ())
                )
                or field_type is type(None)
            )
            if not is_already_optional:
                field_type = Optional[field_type]

        try:
            field_info = PydanticField(
                default=field_default,
                description=prop_schema.get("description"),
                title=prop_schema.get("title"),
                # TODO: Add more mappings as needed based on supported schema keywords
            )
        except TypeError as field_exc:
            raise InvalidSchemaStructureError(
                structure_element="schema constraints",
                expected_type="valid Pydantic Field arguments",
                actual_type=prop_schema,
                context=f"Property '{prop_name_hint}' - {field_exc}",
            ) from field_exc

        logger.debug(
            f"Processed schema for '{prop_name_hint}': Type={field_type}, Required={is_required}, Default={field_default}"
        )
        return (field_type, field_info)

    def _generate_unique_model_name(self, base_name: str) -> str:
        """Generates a unique Python identifier (CamelCase) for a model name.

        Handles sanitization, capitalization, and avoids collisions with previously
        generated names within the current processing context.

        Args:
            base_name: A descriptive base name, often derived from property names or $ref names.

        Returns:
            A unique, valid Python class name string.
        """
        sanitized_base = sanitize_name(base_name)
        capitalized_base = "".join(
            word.capitalize() for word in sanitized_base.split("_") if word
        )
        if not capitalized_base:
            capitalized_base = "Unnamed"

        model_name = f"{capitalized_base}Model"
        count = 1
        while (
            model_name in self._created_models_cache
            or model_name in self._processing_stack
        ):
            model_name = f"{capitalized_base}Model{count}"
            count += 1
        return model_name

    def _create_model_from_schema(
        self,
        model_name: str,
        schema: Dict[str, Any],
        base_model: Type[PydanticBaseModel],
        definitions: Dict[str, Any],
    ) -> Type[PydanticBaseModel]:
        """
        Creates a single Pydantic model class object from its schema definition.

        This function orchestrates the processing of schema 'properties', handling
        recursion for nested objects/refs, and ultimately calls `pydantic.create_model`.

        Args:
            model_name: The desired (and unique) Python class name for the model.
            schema: The JSON schema dictionary for the object to be modeled.
            base_model: The Pydantic base model class to inherit from.
            definitions: Global schema definitions dictionary for resolving `$ref`.

        Returns:
            The dynamically created Pydantic model class object.

        Raises:
            SchemaCircularReferenceError: If this model name is already in the processing stack.
            InvalidSchemaStructureError: If the schema's 'properties' key is invalid.
            SchemaError: (Or subclasses) If any error occurs during property processing
                         via `_schema_to_pydantic_field`.
            Any Exception raised by `pydantic.create_model` if the final field definitions are invalid.
        """
        if model_name in self._created_models_cache:
            return self._created_models_cache[model_name]

        if model_name in self._processing_stack:
            logger.warning(
                f"Circular reference detected for model '{model_name}'. Using ForwardRef."
            )
            forward_ref_type = ForwardRef(f"'{model_name}'")
            self._created_models_cache[model_name] = forward_ref_type
            return forward_ref_type

        logger.info(f"Creating Pydantic model object: '{model_name}'")
        self._processing_stack.append(model_name)

        try:
            fields: Dict[str, PydanticFieldInfo] = {}
            required_props = set(schema.get("required", []))
            properties = schema.get("properties", {})

            if not isinstance(properties, dict):
                raise InvalidSchemaStructureError(
                    structure_element="properties",
                    expected_type="dict",
                    actual_type=properties,
                    context=f"Model '{model_name}'",
                )

            # Process each property defined in the schema
            for orig_prop_name, prop_schema in properties.items():
                py_prop_name = sanitize_name(orig_prop_name)
                if not py_prop_name:
                    logger.warning(
                        f"Skipping property with invalid sanitized name from '{orig_prop_name}' in model '{model_name}'"
                    )
                    continue

                # Recursively determine the Python type and Pydantic Field config for this property
                field_type, field_info = self._schema_to_pydantic_field(
                    prop_schema=prop_schema,
                    prop_name_hint=f"{model_name}_{py_prop_name}",
                    definitions=definitions,
                    base_model=base_model,
                    is_required=(orig_prop_name in required_props),
                )

                # Set alias on the Field object if the Python name differs from the original JSON name
                if py_prop_name != orig_prop_name:
                    field_info.alias = orig_prop_name
                    field_info.validation_alias = orig_prop_name
                    field_info.serialization_alias = orig_prop_name

                # Handle potential collisions if multiple original names sanitize to the same Python name
                final_py_prop_name = py_prop_name
                count = 1
                while final_py_prop_name in fields:
                    final_py_prop_name = f"{py_prop_name}_{count}"
                    count += 1
                    # Ensure the aliased field still points to the original name
                    field_info.alias = orig_prop_name
                    field_info.validation_alias = orig_prop_name
                    field_info.serialization_alias = orig_prop_name
                    logger.warning(
                        f"Sanitized name collision for '{orig_prop_name}' in model '{model_name}'. Using '{final_py_prop_name}' with alias."
                    )

                logger.debug(
                    f"[{model_name}] Mapping field: original='{orig_prop_name}', python='{final_py_prop_name}', type='{field_type}', required={orig_prop_name in required_props}, alias='{field_info.alias}'"
                )
                fields[final_py_prop_name] = (field_type, field_info)

            # Dynamically create the Pydantic model class using the collected field definitions
            new_model = create_model(model_name, __base__=base_model, **fields)

            # Cache the successfully created model object (replace ForwardRef if it was there)
            self._created_models_cache[model_name] = new_model
            logger.info(f"Successfully created Pydantic model: '{model_name}'")
            return new_model

        except SchemaError:
            logger.error(
                f"Schema error during creation of Pydantic model '{model_name}'",
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(
                f"Failed during creation of Pydantic model '{model_name}': {type(e).__name__} - {e}",
                exc_info=True,
            )
            raise SchemaError(
                f"Unexpected failure creating model '{model_name}'"
            ) from e

        finally:
            if self._processing_stack and self._processing_stack[-1] == model_name:
                self._processing_stack.pop()

    def process_tool_schema(
        self,
        tool_name: str,
        input_schema: Dict[str, Any],
        definitions: Dict[str, Any] = {},
        base_model: Type[PydanticBaseModel] = PydanticBaseModel,
    ) -> Tuple[Type[PydanticBaseModel], Dict[str, str]]:
        """
        Processes the top-level input JSON schema for a specific tool.

        This is the main entry point for converting a tool's input schema into a
        usable Pydantic model object. It orchestrates the model creation via
        `_create_model_from_schema` and generates a mapping between the final
        Python field names and the original JSON property names.

        Args:
            tool_name: The logical name of the tool (used for generating model names).
            input_schema: The JSON schema object describing the tool's expected input.
                          Must be of `type: object`.
            definitions: Optional dictionary containing shared schema definitions that can be
                         referenced via `$ref` from within `input_schema` or nested schemas.
            base_model: Optional base Pydantic model class for the generated model to inherit from.
                        Defaults to `pydantic.BaseModel`.

        Returns:
            A tuple containing:
            - `main_input_model`: The dynamically generated Pydantic model class object representing the tool's input.
            - `param_map`: A dictionary mapping the sanitized Python field names used in the
                           generated model to the original JSON property names from the schema.
                           Useful if the caller needs to map arguments back.

        Raises:
            InvalidSchemaTypeError: If `input_schema` is not a dictionary or not `type: object`.
            SchemaError: (Or subclasses) If any error occurs during schema processing or model creation.
        """
        if not isinstance(input_schema, dict) or input_schema.get("type") != "object":
            raise InvalidSchemaTypeError(
                expected_type="object schema dictionary",
                actual_type=input_schema,
                context=f"Tool '{tool_name}' input",
            )

        main_model_name = self._generate_unique_model_name(f"{tool_name}_Input")

        # Create the main Pydantic model object
        main_input_model = self._create_model_from_schema(
            model_name=main_model_name,
            schema=input_schema,
            base_model=base_model,
            definitions=definitions,
        )

        param_map: Dict[str, str] = {}
        model_fields = getattr(main_input_model, 'model_fields', {})

        for field_py_name, field_obj in model_fields.items():
            field_alias = getattr(field_obj, 'alias', None)

            original_name = field_alias if field_alias is not None else field_py_name
            param_map[field_py_name] = original_name

        logger.debug(f"Final param_map for {main_model_name}: {param_map}")

        properties = input_schema.get("properties", {})
        if isinstance(properties, dict):
            mapped_original_names = set(param_map.values())
            for orig_name_from_schema in properties.keys():
                if orig_name_from_schema not in mapped_original_names:
                    logger.warning(
                        f"Property '{orig_name_from_schema}' from input schema was not found "
                        f"as a field or alias value in the final model '{main_model_name}'. "
                        f"Was it skipped or renamed unexpectedly?")

        return main_input_model, param_map

    def get_created_models(self) -> Dict[str, Type[PydanticBaseModel]]:
        """Returns a copy of the internal cache of all models created during the process.

        Useful for debugging or for performing operations like resolving ForwardRefs
        after all potentially related models have been processed.
        """
        return self._created_models_cache.copy()
