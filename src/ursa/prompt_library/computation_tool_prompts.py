code_schema_prompt = """
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CodeExecutionDescriptor",
  "type": "object",
  "properties": {
    "code": {
      "type": "object",
      "description": "Details about the code to run.",
      "properties": {
        "name": {
          "type": "string",
          "description": "The name or identifier of the code/script to run."
        },
        "options": {
          "type": "object",
          "description": "A set of key-value options or parameters for code execution.",
          "additionalProperties": {
            "type": ["string", "number", "boolean"]
          }
        }
      },
      "required": ["name"]
    },
    "inputs": {
      "type": "array",
      "description": "List of input parameters with names and descriptions.",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "Name of the input parameter."
          },
          "description": {
            "type": "string",
            "description": "Description of the input parameter."
          }
        },
        "required": ["name", "description"]
      }
    },
    "outputs": {
      "type": "array",
      "description": "List of expected outputs with names and descriptions.",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "Name of the output value."
          },
          "description": {
            "type": "string",
            "description": "Description of what the output represents."
          }
        },
        "required": ["name", "description"]
      }
    },
  },
  "required": ["code", "inputs", "outputs"]
}
"""