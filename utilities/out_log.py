import sys

def print_to_file(file, text):
    with open(file, 'a') as f:
        f.write(text)
        f.write('\n')