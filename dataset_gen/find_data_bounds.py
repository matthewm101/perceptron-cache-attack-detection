# This file goes through the folder of raw samples an computes the max value of each datapoint.
# It also looks for useless datapoints (same value across all samples) and removes them from the final dataset.

import bz2
import glob
import os
import pickle


found_datapoints = set()    # The set of all datapoint names.
mins = {}   # A mapping from each datapoint name to the minimum value found for that datapoint across all samples.
maxes = {}  # Same as above, but max.
is_constant = set() # The set of all datapoint names that are known to have constant values. When a sample proves that a datapoint is not constant, it is removed from this set.
constants = {}  # A mapping from each datapoint name that is assumed to be constant to the value that the datapoint has.
descs = {}  # A mapping from each datapoint name to its description, if available.
units = {}  # A mapping from each datapoint name to its unit (e.g., seconds), if available.

def reset():
    found_datapoints.clear()
    mins.clear()
    maxes.clear()
    is_constant.clear()
    constants.clear()
    descs.clear()
    units.clear()

# This function recursively scans a dictionary to find all the values.
def rec_check(raw: dict, prefix: str):
    t = raw["type"]
    if t == "Group" or t == "Vector":
        for k,v in raw.items():
            if type(v) == dict:
                rec_check(v, prefix + "." + k)
    elif t == "Scalar":
        v = raw["value"]
        if type(v) != float:
            print(f"Weird type: {type(v)}")
        if prefix in found_datapoints:
            maxes[prefix] = max(maxes[prefix], v)
            mins[prefix] = min(mins[prefix], v)
            if prefix in is_constant:
                if constants[prefix] != v:
                    is_constant.remove(prefix)
                    constants.pop(prefix)
        else:
            found_datapoints.add(prefix)
            maxes[prefix] = v
            mins[prefix] = v
            is_constant.add(prefix)
            constants[prefix] = v
            descs[prefix] = raw["description"]
            units[prefix] = raw["unit"]
    elif t == "Distribution":
        pass
    else:
        print(f"Warning: unhandled type '{t}' found in data")
    

# Goes through an entire sample, looking for maximum values and constant values.
def check_sample(sample):
    rec_check(sample["system"]["cpu"], "cpu")
    rec_check(sample["system"]["l2"], "l2")
    rec_check(sample["system"]["mem_ctrls"], "mem_ctrls")
    rec_check(sample["system"]["membus"], "membus")
    rec_check(sample["system"]["tol2bus"], "tol2bus")

# A description of a sample file.
# Sample files are expected to have the following filename format: name_first-last_XXXus.pbz2
# 'name' is the source name, like astar or bzip2. 
# 'first-last' gives the range of sample numbers in the file (for example, 200-299 includes 100 samples).
# 'XXXus' is the length of the samples. For example, '100us' states that each sample represents a period of 100 microseconds.
class SampleFile():
    def __init__(self, fname):
        self.fname = fname
        self.full_sample_name = str(os.path.splitext(os.path.split(self.fname)[1])[0])
        splits = self.full_sample_name.split("_")
        self.source_name = splits[0]
        sno_splits = splits[1].split("-")
        self.first_sample_number = int(sno_splits[0])
        self.last_sample_number = int(sno_splits[1])
        self.n_samples = self.last_sample_number + 1 - self.first_sample_number
        self.sample_length_us = int(splits[2][:-2])

# Opens a file, loads the data, and recursively scans through it to update the sets and dictionaries at the top of this program file.
def handle_file(filename):
    with bz2.open(filename) as f:
        raw_data = pickle.load(f)
    for sample in raw_data:
        check_sample(sample)

# By default, every single file in the raw_samples folder is checked.
# Change the following line if you want to draw samples from somewhere else. Don't forget to change it in generate_dataset.py as well!
filenames = glob.glob("../raw_samples/*.pbz2")
filenames.sort()
files = [SampleFile(n) for n in filenames]
sample_lengths = list(set([f.sample_length_us for f in files]))
sample_lengths.sort()

# If multiple raw sample files with different sample lengths are present in the raw_samples folder, then separate config files are created for each.
for l in sample_lengths:
    reset()
    print(f"Processing data with {l}us sample periods...")
    filtered_files = list(filter(lambda f: f.sample_length_us == l, files))
    source_names = list(set([f.source_name for f in filtered_files]))
    for f in filtered_files:
        print(f"\t{f.full_sample_name}...", end="", flush=True)
        handle_file(f.fname)
        print("done.")
    useful_datapoints = found_datapoints.difference(is_constant)
    print("Outputting config file to the datasets folder...", end="", flush=True)

    # By default, config files are output to the datasets folder.
    # If you wish to output the config files elsewhere, modify this as desired.
    dout = open(f"../datasets/dataset_config_{l}us.ini", "w")

    dout.write(f"# Attack Detection Super-Dataset Configuration\n")
    dout.write(f"# This config is for samples with the following length: {l} microseconds\n\n")

    dout.write("[Global Settings]\n\n")
    dout.write("# By default, all datapoints will be linearly mapped to the range 0.0-1.0,\n")
    dout.write("# by assigning the minimum known value to 0.0 and the max known to 1.0.\n")
    dout.write("# If you instead want to map the value 0 to 0.0 (effectively replacing the\n")
    dout.write("# minimum value of all datapoints with 0), then set this to 'true'.\n")
    dout.write("force_zero_minimum=false\n\n")

    dout.write(f"[Files]\n# These are the files used to generate this config.\n")
    dout.write(f"# When generating a dataset using this config, only these files will be read.\n")
    dout.write(f"# If you want, you can add or delete files from this list,\n")
    dout.write(f"# but keep in mind that the original list was used to generate the mins/maxes below.\n")
    for f in filtered_files:
        dout.write(f"{f.fname}\n")


    dout.write(f"\n[Mins and Maxes]\n# After reading through the samples, {len(useful_datapoints)} useful datapoints were found.\n")
    dout.write(f"# The following lines contain the min and max known values of each datapoint.\n")
    dout.write("# These can be modified if desired. Later they will be used to clamp all data to a 0.0-1.0 range.\n")
    dout.write("# Also, if you want to remove a datapoint, simply delete its min and max entries below.\n\n")

    # Only datapoints with non-constant values are written to config files.
    for dp in useful_datapoints:
        if len(units[dp]) > 0:
            dout.write(f"# Name: {dp}    Unit: {units[dp]}\n")
        else:
            dout.write(f"# Name: {dp}\n")
        if len(descs[dp]) > 0:
            dout.write(f"# Description: {descs[dp]}\n")
        dout.write(f"{dp}.min={mins[dp]}\n")
        dout.write(f"{dp}.max={maxes[dp]}\n\n")
    dout.close()
    print("finished.")
print("All config files generated.")
print("Don't forget to look over the config files in case anything needs to be changed.")    
