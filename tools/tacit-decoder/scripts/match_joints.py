# A joint is where spike model decides to produce a packet
# This script matches the joints from the spike trace to the decoder dump

import argparse
import re

def read_reference_file(reference_file):
    """
    Reads the reference file and returns a list of joint numbers in decimal.
    """
    joint_pattern = r'\[joint\]\s*([0-9a-fA-F]+)'
    with open(reference_file, 'r') as f:
        reference_numbers = []
        for line in f:
            match = re.search(joint_pattern, line)
            if match:
                reference_numbers.append(match.group(1))
    return reference_numbers

def read_created_file(created_file):
    """
    Reads the created file and returns a list of pc numbers in decimal.
    """
    pc_pattern = r'pc:\s*([0-9a-fA-F]+)'
    with open(created_file, 'r') as f:
        created_data = []
        for line in f:
            match = re.search(pc_pattern, line)
            if match:
                created_data.append(match.group(1))
    return created_data

def find_most_recent_divergence(reference_numbers, created_data):
    """
    Finds the most recent divergence between the reference and created data.
    """
    count = 0
    last_matching_address = None
    
    for i,joint in enumerate(reference_numbers):
        count += 1
        if i >= len(created_data):
            print(f"Most recent divergence found at joint: {joint}")
            return joint, None, count
        if created_data[i] != joint:
            print(f"Most recent divergence found at joint: {joint}")
            return joint, created_data[i], count
    print("No divergence found. All joints match.")
    return None, None, count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find divergence in trace.")
    parser.add_argument("-r", "--ref_file", type=str, required=True, help="Path to the reference trace file.")
    parser.add_argument("-d", "--decoder_dump", type=str, required=True, help="Path to the decoder dump file.")
    args = parser.parse_args()

    reference_file = args.ref_file
    decoder_dump = args.decoder_dump

    reference_numbers = read_reference_file(reference_file)
    created_data = read_created_file(decoder_dump)
    
    last_match, divergence, count = find_most_recent_divergence(reference_numbers, created_data)
    
    if divergence:
        print(f"Most recent reference joint: {last_match}")
        print(f"First divergence: {divergence}")
        print(f"At line count: {count}")