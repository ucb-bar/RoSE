# Parse the spike trace and the decoder dump to find any divergence

import argparse

def read_reference_file(reference_file):
    """
    Reads the reference file and returns a list of addresses.
    """
    with open(reference_file, 'r') as f:
        reference_addresses = [line.strip().split(',')[0] for line in f.readlines()]
    return reference_addresses

def read_created_file(created_file):
    """
    Reads the created file and returns a dictionary of addresses and their corresponding lines.
    """
    created_data = {}
    with open(created_file, 'r') as f:
        for line in f:
            if ':' in line and 'timestamp' not in line:
                address, instruction = line.split(':', 1)
                created_data[address.strip()] = instruction.strip()
    return created_data

def find_most_recent_divergence(reference_addresses, created_data):
    """
    Finds the most recent divergence between the reference and created data.
    """
    # first, get the first line of the created file
    first_line_address = list(created_data.keys())[0]
    # get the first occurrence of the first line address in the reference file
    first_line_index = reference_addresses.index(first_line_address.split('0x')[1])
    
    print(f"First line index: {first_line_index}")

    count = 0
    last_matching_address = None
    # start from first_line_index and go through the reference file
    
    for address in reference_addresses[first_line_index:]:
        # Convert address to the format used in the created file
        formatted_address = '0x' + address.lower()

        if formatted_address in created_data:
            last_matching_address = formatted_address
            count += 1
        else:
            print(f"Most recent divergence found at address: {formatted_address}")
            return last_matching_address, formatted_address, count

    print("No divergence found. All addresses match.")
    return last_matching_address, None, count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find divergence in trace.")
    parser.add_argument("-r", "--ref_file", type=str, required=True, help="Path to the reference trace file.")
    parser.add_argument("-d", "--decoder_dump", type=str, required=True, help="Path to the decoder dump file.")
    args = parser.parse_args()

    reference_file = args.ref_file
    decoder_dump = args.decoder_dump

    reference_addresses = read_reference_file(reference_file)
    created_data = read_created_file(decoder_dump)
    
    last_match, divergence, count = find_most_recent_divergence(reference_addresses, created_data)
    
    if divergence:
        print(f"Most recent match: {last_match}")
        print(f"First divergence: {divergence}")
        print(f"At line count: {count}")
    else:
        print(f"Everything matches up to the last reference address: {last_match}")
