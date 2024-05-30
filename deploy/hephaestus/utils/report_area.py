import re
import os
import functools
import argparse

def parse_genus_report(report_text, target_modules):
    # Define regex patterns for parsing
    module_pattern = re.compile(r'^\s*(\S+)\s+(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$')
    top_pattern = re.compile   (r'^\s*(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$')
    target_modules = set(target_modules)

    # Initialize dictionary to store area breakdown
    area_breakdown = {module: {"Cell Count": 0, "Cell Area": 0.0, "Net Area": 0.0, "Total Area": 0.0} for module in target_modules}
    top_module_area_breakdown = {"name": "", "Cell Count": 0, "Cell Area": 0.0, "Net Area": 0.0, "Total Area": 0.0}

    # Parse the report
    first_line = True
    for line in report_text.splitlines():
        if first_line:
            match = top_pattern.match(line)
            if match: 
                top_module, cell_count, cell_area, net_area, total_area = match.groups()
                top_module_area_breakdown["name"] = top_module
                top_module_area_breakdown["Cell Count"] = int(cell_count)
                top_module_area_breakdown["Cell Area"] = float(cell_area)
                top_module_area_breakdown["Net Area"] = float(net_area)
                top_module_area_breakdown["Total Area"] = float(total_area)
                first_line = False

        else:
            match = module_pattern.match(line)
            if match:
                instance, module, cell_count, cell_area, net_area, total_area = match.groups()
                if functools.reduce(lambda x,y : x or y, [i in module for i in target_modules]):
                    module = module.split("_")[0]
                    area_breakdown[module]["Cell Count"] += int(cell_count)
                    area_breakdown[module]["Cell Area"] += float(cell_area)
                    area_breakdown[module]["Net Area"] += float(net_area)
                    area_breakdown[module]["Total Area"] += float(total_area)

    return top_module_area_breakdown, area_breakdown

# Example usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, help='The path to the report file', default='')
    args = parser.parse_args()

    report_file_path = '../../../soc/sim/firesim/target-design/chipyard/vlsi/build'
    config_name = args.target
    report_file_path = os.path.join(report_file_path, config_name, 'syn-rundir/reports/final_area.rpt')

    report_text = open(report_file_path).read()

    target_modules = ["SerialWidthAggregator", "SRAMImgBufferExcess", "SADPipe"]
    top_module_area_breakdown, area_breakdown = parse_genus_report(report_text, target_modules)

    print(f"Top Module: {top_module_area_breakdown['name']}")
    # print(f"  Cell Count: {top_module_area_breakdown['Cell Count']}")
    # print(f"  Cell Area: {top_module_area_breakdown['Cell Area']}")
    # print(f"  Net Area: {top_module_area_breakdown['Net Area']}")
    print(f"  Total Area: {top_module_area_breakdown['Total Area']}")
    print()

    for module, breakdown in area_breakdown.items():
        print(f"Module: {module}")
        # print(f"  Cell Count: {breakdown['Cell Count']}")
        # print(f"  Cell Area: {breakdown['Cell Area']}")
        # print(f"  Net Area: {breakdown['Net Area']}")
        print(f"  Total Area: {breakdown['Total Area']}")
        print()

