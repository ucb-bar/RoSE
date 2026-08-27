# given a file, filter out the lines that start with [bp]:

import sys

def filter_bp(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('[bp]'):
                print(line)

if __name__ == '__main__':
    filter_bp(sys.argv[1])