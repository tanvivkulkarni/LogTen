import os

def replace_in_csv_files(folder_path):
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)

            # Read file content
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Replace ':' with '.'
            updated_content = content.replace(":", ".")

            # Write back to the same file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(updated_content)

            print(f"Updated: {filename}")

# 👉 Replace this with your folder path
folder_path = "output_folder/27th_folder/LG1_6"

replace_in_csv_files(folder_path)