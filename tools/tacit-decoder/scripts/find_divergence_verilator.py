# Parse the verilator trace and the decoder dump to find any divergence

import argparse
import re
def read_reference_file(reference_file):
    """
    Reads the reference file and returns a list of addresses.
    """
    with open(reference_file, 'r') as f:
        reference_addresses = [line.strip() for line in f.readlines()]
    # print(f"len(reference_addresses): {len(reference_addresses)}")
    return reference_addresses

def get_start_time(file):
    with open(file, 'r') as f:
        # get the first line
        first_line = f.readline()
        # [timestamp: 115213] Start
        pattern = r"\[timestamp: (\d+)\]"
        # print(f"first_line: {first_line}")
        timestamp = int(re.search(pattern, first_line).group(1))
        return timestamp

def find_start_of_trace(reference_addresses, start_time):
    """
    Finds the line number containing the start time
    """
    for i, line in enumerate(reference_addresses):
        if str(start_time) in line:
            return i
    return None

def parse_verilator_trace(reference_addresses, start_of_trace_index, length, target_address):
    """
    Parses the verilator trace and returns a dictionary of addresses and their corresponding lines.
    """
    # pattern: 
    # C0:         19 [1] pc=[0000000000010000] W[r10=0000000000010000][1] R[r 0=0000000000000000] R[r 0=0000000000000000] inst=[00000517] auipc   a0, 0x0
    ref_data = []

    extracted_pc = lambda line: line.split('pc=[')[1].split(']')[0]
    # find the offset
    offset = 0
    while True:
        try:
            ref_pc = extracted_pc(reference_addresses[start_of_trace_index - 1 + offset])
            # print(f"ref_pc: {ref_pc}")
            # print(f"target_address: {target_address}")
        except:
            # print(f"start_of_trace_index: {start_of_trace_index - 1 + offset}")
            # print(f"reference_addresses: {reference_addresses[start_of_trace_index - 1 + offset]}")
            offset += 1
            continue
        if target_address in ref_pc:
            break
        offset += 1

    start = start_of_trace_index + offset - 1
    end = start + length + 1
    
    for i in range(start, end):
        try:
            ref_data.append(extracted_pc(reference_addresses[i]))
        except:
            continue
    return ref_data


def read_created_file(created_file):
    """
    Reads the created file and returns a dictionary of addresses and their corresponding lines.
    """
    created_data = []
    with open(created_file, 'r') as f:
        for line in f:
            if ':' in line and 'timestamp' not in line and "hit count" not in line:
                address, instruction = line.split(':', 1)
                created_data.append(address.strip().split('0x')[1])
    # print(len(created_data))
    return created_data

def find_most_recent_divergence(reference_addresses, created_data, start_time):
    """
    Finds the most recent divergence between the reference and created data.
    """
    count = 0
    last_matching_address = None

    target_address = created_data[0]

    start_of_trace_index = find_start_of_trace(reference_addresses, start_time)
    print(f"start_of_trace_index: {start_of_trace_index}")
    ref_data = parse_verilator_trace(reference_addresses, start_of_trace_index, len(created_data), target_address)

    # matching the created data to the reference data
    # print(len(ref_data))
    # print(len(created_data))
    for i, address in enumerate(created_data):
        if address in ref_data[i]:
            count += 1
        else:
            print(f"Most recent divergence found at address: {address}")
            print(f"reference data: {ref_data[i]}")
            print(f"created data: {address}")
            return False, count
    return True, count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find divergence in trace.")
    parser.add_argument("-r", "--ref_file", type=str, required=True, help="Path to the reference trace file.")
    parser.add_argument("-d", "--decoder_dump", type=str, required=True, help="Path to the decoder dump file.")
    args = parser.parse_args()

    reference_file = args.ref_file
    decoder_dump = args.decoder_dump

    reference_addresses = read_reference_file(reference_file)
    created_data = read_created_file(decoder_dump)
    start_time = get_start_time(decoder_dump)
    match, count = find_most_recent_divergence(reference_addresses, created_data, start_time)
    if match:
        print(f"[SUCCESS] Everything matched up to the last line: {count}!")
    else:
        print(f"[FAILURE] Divergence found at line: {count}!")
