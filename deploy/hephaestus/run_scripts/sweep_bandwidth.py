#!/usr/bin/env python3
import os
import yaml

soc_freq = 1e9
target_bw = [5e7, 1e8, 2e8, 4e8, 8e8, 16e8]
json_path = "../../config/config_gym_MiddleBuryEnv-v0.yaml"
target_channel = 1

if __name__ == "__main__":
    config_bw = [int(4 * soc_freq // bw) for bw in target_bw]
    print("Computed bandwidth configurations: ", config_bw)

    for bw in config_bw:
        with open (json_path, "r") as f:
            config = yaml.safe_load(f)
            for list in config["channel_bandwidth"]:
                list[target_channel] = bw
        # write back to a new file
        cwd = os.getcwd()
        new_json_path = os.path.join(cwd, "intermediate", "config_gym_MiddleBuryEnv-v0_bw" + str(bw) + ".yaml")
        print("Writing new config to: ", new_json_path)
        with open(new_json_path, "w") as f:
            yaml.dump(config, f)
            
