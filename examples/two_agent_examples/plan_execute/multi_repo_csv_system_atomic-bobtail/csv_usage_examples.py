# Examples of usage for CSV utility functions in utils_lib

# Example 1: Reading a CSV file
# Assume we have a CSV file 'data.csv' with the following content:
# name,age,city
# Alice,30,New York
# Bob,25,Los Angeles
# Charlie,35,Chicago

# Read the CSV file:
# data = read_csv('data.csv')
# print(data)  # Output: [{'name': 'Alice', 'age': '30', 'city': 'New York'}, {'name': 'Bob', 'age': '25', 'city': 'Los Angeles'}, {'name': 'Charlie', 'age': '35', 'city': 'Chicago'}]

# Example 2: Writing to a CSV file
# data_to_write = [{'name': 'David', 'age': '28', 'city': 'San Francisco'}]
# write_csv('output.csv', data_to_write)

# Example 3: Filtering CSV data
# Assume the data variable contains the data read from 'data.csv'
# Filter function to select only people aged 30 or more:
# filtered_data = filter_csv(data, lambda x: int(x['age']) >= 30)
# print(filtered_data)  # Output: [{'name': 'Alice', 'age': '30', 'city': 'New York'}, {'name': 'Charlie', 'age': '35', 'city': 'Chicago'}]

# Example 4: Merging CSV files
# file1.csv content:
# name,age,city
# Alice,30,New York
# file2.csv content:
# name,age,city
# Bob,25,Los Angeles
#
# Merge 'file1.csv' and 'file2.csv' into 'merged.csv'
# merge_csv(['file1.csv', 'file2.csv'], 'merged.csv')
# Merged data will be written into 'merged.csv'
