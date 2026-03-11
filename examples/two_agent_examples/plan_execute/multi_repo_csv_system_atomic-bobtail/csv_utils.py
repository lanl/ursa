import csv
from typing import Any


def read_csv(file_path: str) -> list[dict[str, Any]]:
    """
    Reads a CSV file and returns a list of dictionaries. Each dictionary represents a row with column headers as keys.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the rows in the CSV file.
    """
    with open(file_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [row for row in reader]


def write_csv(file_path: str, data: list[dict[str, Any]]) -> None:
    """
    Writes a list of dictionaries to a CSV file.

    Args:
        file_path (str): The path where the CSV file will be written.
        data (List[Dict[str, Any]]): A list of dictionaries representing the rows to write.
    """
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        if data:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)


def filter_csv(
    data: list[dict[str, Any]], filter_func: callable
) -> list[dict[str, Any]]:
    """
    Filters a list of dictionaries based on a filter function.

    Args:
        data (List[Dict[str, Any]]): The list of dictionaries to filter.
        filter_func (callable): A function that takes a dictionary and returns True if it should be included.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries that passed the filter.
    """
    return [row for row in data if filter_func(row)]


def merge_csv(file_paths: list[str], output_path: str) -> None:
    """
    Merges multiple CSV files into a single CSV file.

    Args:
        file_paths (List[str]): The list of file paths to merge.
        output_path (str): The path where the merged CSV file will be written.
    """
    merged_data = []
    for file_path in file_paths:
        merged_data.extend(read_csv(file_path))
    write_csv(output_path, merged_data)
