import time

def follow(file_path):
    with open(file_path, 'r') as file:
        # Go to the end of the file
        file.seek(0, 2)
        
        while True:
            line = file.readline()
            if line:
                print(line, end='')  # Print the new line
            else:
                time.sleep(0.5)  # Wait for new content

# Call the function with the path to your stdout file
follow('stdout.out')

