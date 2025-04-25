import os

def load_files_to_dict(directory, extension=".py"):
    file_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            path = os.path.join(directory, filename) 
            with open(path, 'r') as f: 
                file_dict[filename] = f.read()

    return file_dict

