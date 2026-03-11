import argparse

import csv_utils


def main():
    parser = argparse.ArgumentParser(description="CSV Command Line Interface")
    parser.add_argument(
        "command", choices=["filter", "merge"], help="Command to execute"
    )
    parser.add_argument("--input", nargs="+", help="Input CSV file(s)")
    parser.add_argument("--output", help="Output CSV file")
    parser.add_argument(
        "--filter", help="Filter function (Python code as a string)"
    )

    args = parser.parse_args()

    if args.command == "filter" and args.input and args.output:
        data = csv_utils.read_csv(args.input[0])
        # Example filter function from string (this will need eval or ast.literal_eval)
        filter_func = eval("lambda row: " + args.filter)
        filtered_data = csv_utils.filter_csv(data, filter_func)
        csv_utils.write_csv(args.output, filtered_data)
        print(f"Filtered data written to {args.output}")

    elif args.command == "merge" and args.input and args.output:
        csv_utils.merge_csv(args.input, args.output)
        print(f"Merged data written to {args.output}")


if __name__ == "__main__":
    main()
