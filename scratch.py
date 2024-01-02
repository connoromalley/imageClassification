import os

folder_path = "tracks_data/deer"

# List all items (files and directories) in the specified folder
items = os.listdir(folder_path)

# Count the number of items in the folder
num_items = len(items)

print(f"There are {num_items} items in the folder.")
