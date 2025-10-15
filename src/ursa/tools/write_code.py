import os

from langchain_core.tools import tool


@tool
def write_python(code: str, filename: str, workspace_dir: str) -> str:
    """
    Writes code to a file in the given workspace.

    Args:
        code: The code to write
        filename: the filename to write

    Returns:
        File writing status: string
    """
    print("Writing filename ", filename)
    try:
        # Extract code if wrapped in markdown code blocks
        if "```" in code:
            code_parts = code.split("```")
            if len(code_parts) >= 3:
                # Extract the actual code
                if "\n" in code_parts[1]:
                    code = "\n".join(code_parts[1].strip().split("\n")[1:])
                else:
                    code = code_parts[2].strip()

        # Write code to a file
        code_file = os.path.join(workspace_dir, filename)

        with open(code_file, "w") as f:
            f.write(code)
        print(f"Written code to file: {code_file}")

        return f"File {filename} written successfully."

    except Exception as e:
        print(f"Error generating code: {str(e)}")
        # Return minimal code that prints the error
        return f"Failed to write {filename} successfully."
