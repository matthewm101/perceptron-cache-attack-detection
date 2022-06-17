# This file contains the configuration used by run_simulation.py.

from common import Options
import argparse
import os
from m5.objects import *
from m5.params import String

# This is just used to get the git root to make the paths easier.
# You can either do "pip install gitpython", or you can replace SPEC_PATH, CUSTOM_PATH, and OUTPUT_DIR with absolute paths, making these lines optional.
import git
git_root = git.Repo(".",search_parent_directories=True).working_tree_dir

###########################################
# SETUP PARAMETERS                        #
# Modify these to match your file system. #
###########################################

# The path to your SPEC2006 directory. Make sure the path ends with a slash.
SPEC_PATH = git_root + "/SPEC2006/"
# The path from the SPEC2006 root to the benchmark directory. This is probably already correct.
RUN_DIR_prefix  = 'benchspec/CPU2006/'
# The location of the executable of a benchmark relative to the benchmark's directory.
# Make sure this matches your SPEC2006 directory, and make sure it starts and ends with a slash.
RUN_DIR_suffix = '/run/run_base_ref_amd64-m64-gcc41-nn.0000/'
# The suffix appended to directories in the RUN_DIR_prefix directory. Make sure this matches.
x86_suffix = '_base.amd64-m64-gcc41-nn'

# The path to the custom benchmark folder. Make sure the path ends with a slash.
CUSTOM_PATH = git_root + "/custom_benchmarks/"

# Where raw samples should be output. Change this to your preferred location.
OUTPUT_DIR = git_root + "/raw_samples/"

##########################################################
# SIMULATOR PARAMETERS                                   #
# Modify these to control how raw samples are collected. #
##########################################################

# All times are written as seconds; times can be floating-point numbers.

# The time skipped past at the start of a simulation (without saving samples).
# This is mostly useful for ensuring that benchmarks clear their starting code.
SKIPPED_TIME = 1e-2

# All the different sampling rates used. Make sure this list is sorted. Also, periods should be multiples of 1e-6.
# When multiple sample rates are provided, the following occurs:
#   The first period of time is simulated, then the debug counters are dumped for the first sampling rate.
#   The time difference between the first and second periods of time is then simulated, and counters are dumped for the second rate.
#   Then the time difference between the second and third periods is simulated and dumped
#   This repeats until the diff between the 2nd to last and last periods is simulated, with the counters being reset afterwards.
#   So, if the sampling periods are [1e-6, 1e-5, 1e-4], the following occurs:
#       1e-6 seconds are simulated, then counters are dumped for 1us.
#       1e-5 - 1e-6 = 9e-6 seconds are simulated (making the total 1e-5), then counters are dumped for 10us.
#       1e-4 - 1e-5 = 9e-5 seconds are simulated (making the total 1e-4), then counters are dumped for 100us.
#       Counters are reset and the cycle restarts.
#   An important thing to remember is that only the samples for the largest rate are truly contiguous in time;
#   any smaller sample rates included will have gaps between consecutive samples.
# If you need samples to be contiguous in time or want the simulation to run faster, then this should be a single-value list.
# Otherwise, you can include multiple periods to easily generate samples that capture smaller amounts of time (but are not contigious).
SAMPLING_PERIODS = [1e-5, 1e-4]

# The number of sample batches to make, and the number of samples per batch.
# Each batch is saved after all samples in the batch are recorded.
# Larger batches take longer to save, although they may benefit from better compression.
BATCHES_PER_BENCHMARK = 1
SAMPLES_PER_BATCH = 100

# The system is set up using gem5's built-in configurations.
# The options below can be modified to test slightly different systems.
parser = argparse.ArgumentParser()
Options.addCommonOptions(parser)
Options.addSEOptions(parser)
options = parser.parse_args([
    "--num-cpus=1", "--mem-size=2GB", "--mem-type=DDR4_2400_8x8",
    "--cpu-type", "DerivO3CPU",
    "--sys-clock=1GHz", "--cpu-clock=2GHz", # these are the defaults
    "--caches", "--l2cache",
    "--l1d_size=64kB", "--l1i_size=32kB", "--l2_size=2MB",
    "--l1d_assoc=8",  "--l1i_assoc=4", "--l2_assoc=8",
    "--cacheline_size=64", # default
])

#############################################################
# BENCHMARK PARAMETERS                                      #
# Modify these to control how the benchmarks are simulated. #
#############################################################

# To add a SPEC benchmark, register it using the SPECBenchmarkBuilder class below. Note that some benchmarks might not fit the class definition in its current state.

# To add a standalone executable, register it using the CustomBenchmarkBuilder class instead.

benchmap = {}
class SPECBenchmarkBuilder:
    def __init__(self, name: str):
        self.process = Process()
        self.cwd = SPEC_PATH + RUN_DIR_prefix + name + RUN_DIR_suffix
        self.process.executable = String(self.cwd + name.split(".")[1] + x86_suffix)
        self.process.cmd = [self.process.executable]
        self.process.cwd = self.cwd
        benchmap[name.split(".")[1]] = self.process
    def arg(self, arg: str):
        self.process.cmd.append(String(arg))
        return self
    def file(self, file: str):
        self.process.cmd.append(String(self.cwd + file))
        return self
    def input(self, input: str):
        self.process.input = String(input)
        return self
    def finput(self, input: str):
        self.process.input = String(self.cwd + input)
        return self

# To set up a benchmark, first call SPECBenchmarkBuilder with the full name of the benchmark (number.name).
# Command line arguments can be added by calling .arg() with the name of the arg inside.
# If the argument is a file in the SPEC executable's directory, use .file() instead.
# To provide an input, use .input() for regular strings, or .finput() for filenames.

# Note that some benchmarks don't follow the same naming conventions and will need their commands constructed differently.
# Also, some benchmarks might not work due to working directory issues.
# Issues with having the right cwd is why I'm using absolute paths everywhere in this file.

SPECBenchmarkBuilder("401.bzip2").file("input.source").arg("10") # Smaller value used to avoid longer startup times
SPECBenchmarkBuilder("403.gcc").file("166.i").arg("-o").file("166.s")
SPECBenchmarkBuilder("429.mcf").file("inp.in")
SPECBenchmarkBuilder("445.gobmk").arg("--quiet").arg("--mode").arg("gtp").finput("13x13.tst")
SPECBenchmarkBuilder("456.hmmer").file("nph3.hmm").file("swiss41")
SPECBenchmarkBuilder("458.sjeng").file("ref.txt")
SPECBenchmarkBuilder("462.libquantum").arg("1397").arg("8")
SPECBenchmarkBuilder("464.h264ref").arg("-d").file("foreman_ref_encoder_baseline.cfg")
SPECBenchmarkBuilder("471.omnetpp").file("omnetpp.ini")
SPECBenchmarkBuilder("473.astar").file("rivers.cfg")


class CustomBenchmarkBuilder:
    def __init__(self, bench_name: str, path: str, exe_name: str):
        self.process = Process()
        self.cwd = CUSTOM_PATH + path
        self.process.executable = String(self.cwd + exe_name)
        self.process.cmd = [self.process.executable]
        self.process.cwd = self.cwd
        benchmap[bench_name] = self.process

    # Use this to add a regular, non-file argument (like a number or flag).
    def arg(self, arg: str):
        self.process.cmd.append(String(arg))
        return self
    
    # Use this to add a file argument.
    # The provided filename should be written relative to the CUSTOM_PATH directory.
    # The filename will be rewritten as an absolute path (to make sure things work properly).
    def file(self, file: str):
        self.process.cmd.append(String(self.cwd + file))
        return self

    # Use this to add raw text piped into stdin.
    def input(self, input: str):
        self.process.input = String(input)
        return self

    # Use this to add a file whose contents are piped into stdin.
    # The same rules as the file function apply.
    def finput(self, input: str):
        self.process.input = String(self.cwd + input)
        return self



CustomBenchmarkBuilder("idletimer", "", "idletimer")
# Threshold for PP: between 26 and 222
# Threshold for FF: between 93 and 95 (note: FF is particularly inconsistent in gem5)
# Threshold for FR: between 26 and 166
CustomBenchmarkBuilder("pp-test-2500", "", "cacheattack").arg("pp").arg("test").arg("2500").arg("120")
CustomBenchmarkBuilder("pp-test-5000", "", "cacheattack").arg("pp").arg("test").arg("5000").arg("120")
CustomBenchmarkBuilder("pp-test-7500", "", "cacheattack").arg("pp").arg("test").arg("7500").arg("120")
CustomBenchmarkBuilder("pp-test-10000", "", "cacheattack").arg("pp").arg("test").arg("10000").arg("120")
CustomBenchmarkBuilder("pp-test-max", "", "cacheattack").arg("pp").arg("test").arg("max").arg("120")
CustomBenchmarkBuilder("pp-tx-2500", "", "cacheattack").arg("pp").arg("tx").arg("2500").arg("120")
CustomBenchmarkBuilder("pp-tx-5000", "", "cacheattack").arg("pp").arg("tx").arg("5000").arg("120")
CustomBenchmarkBuilder("pp-tx-7500", "", "cacheattack").arg("pp").arg("tx").arg("7500").arg("120")
CustomBenchmarkBuilder("pp-tx-10000", "", "cacheattack").arg("pp").arg("tx").arg("10000").arg("120")
CustomBenchmarkBuilder("pp-tx-max", "", "cacheattack").arg("pp").arg("tx").arg("max").arg("120")
CustomBenchmarkBuilder("pp-rx-2500", "", "cacheattack").arg("pp").arg("rx").arg("2500").arg("120")
CustomBenchmarkBuilder("pp-rx-5000", "", "cacheattack").arg("pp").arg("rx").arg("5000").arg("120")
CustomBenchmarkBuilder("pp-rx-7500", "", "cacheattack").arg("pp").arg("rx").arg("7500").arg("120")
CustomBenchmarkBuilder("pp-rx-10000", "", "cacheattack").arg("pp").arg("rx").arg("10000").arg("120")
CustomBenchmarkBuilder("pp-rx-max", "", "cacheattack").arg("pp").arg("rx").arg("max").arg("120")

CustomBenchmarkBuilder("fr-test-2500", "", "cacheattack").arg("fr").arg("test").arg("2500").arg("120")
CustomBenchmarkBuilder("fr-test-5000", "", "cacheattack").arg("fr").arg("test").arg("5000").arg("120")
CustomBenchmarkBuilder("fr-test-7500", "", "cacheattack").arg("fr").arg("test").arg("7500").arg("120")
CustomBenchmarkBuilder("fr-test-10000", "", "cacheattack").arg("fr").arg("test").arg("10000").arg("120")
CustomBenchmarkBuilder("fr-test-max", "", "cacheattack").arg("fr").arg("test").arg("max").arg("120")
CustomBenchmarkBuilder("fr-tx-2500", "", "cacheattack").arg("fr").arg("tx").arg("2500").arg("120")
CustomBenchmarkBuilder("fr-tx-5000", "", "cacheattack").arg("fr").arg("tx").arg("5000").arg("120")
CustomBenchmarkBuilder("fr-tx-7500", "", "cacheattack").arg("fr").arg("tx").arg("7500").arg("120")
CustomBenchmarkBuilder("fr-tx-10000", "", "cacheattack").arg("fr").arg("tx").arg("10000").arg("120")
CustomBenchmarkBuilder("fr-tx-max", "", "cacheattack").arg("fr").arg("tx").arg("max").arg("120")
CustomBenchmarkBuilder("fr-rx-2500", "", "cacheattack").arg("fr").arg("rx").arg("2500").arg("120")
CustomBenchmarkBuilder("fr-rx-5000", "", "cacheattack").arg("fr").arg("rx").arg("5000").arg("120")
CustomBenchmarkBuilder("fr-rx-7500", "", "cacheattack").arg("fr").arg("rx").arg("7500").arg("120")
CustomBenchmarkBuilder("fr-rx-10000", "", "cacheattack").arg("fr").arg("rx").arg("10000").arg("120")
CustomBenchmarkBuilder("fr-rx-max", "", "cacheattack").arg("fr").arg("rx").arg("max").arg("120")

CustomBenchmarkBuilder("ff-test-2500", "", "cacheattack").arg("ff").arg("test").arg("2500").arg("94")
CustomBenchmarkBuilder("ff-test-5000", "", "cacheattack").arg("ff").arg("test").arg("5000").arg("94")
CustomBenchmarkBuilder("ff-test-7500", "", "cacheattack").arg("ff").arg("test").arg("7500").arg("94")
CustomBenchmarkBuilder("ff-test-10000", "", "cacheattack").arg("ff").arg("test").arg("10000").arg("94")
CustomBenchmarkBuilder("ff-test-max", "", "cacheattack").arg("ff").arg("test").arg("max").arg("94")
CustomBenchmarkBuilder("ff-tx-2500", "", "cacheattack").arg("ff").arg("tx").arg("2500").arg("94")
CustomBenchmarkBuilder("ff-tx-5000", "", "cacheattack").arg("ff").arg("tx").arg("5000").arg("94")
CustomBenchmarkBuilder("ff-tx-7500", "", "cacheattack").arg("ff").arg("tx").arg("7500").arg("94")
CustomBenchmarkBuilder("ff-tx-10000", "", "cacheattack").arg("ff").arg("tx").arg("10000").arg("94")
CustomBenchmarkBuilder("ff-tx-max", "", "cacheattack").arg("ff").arg("tx").arg("max").arg("94")
CustomBenchmarkBuilder("ff-rx-2500", "", "cacheattack").arg("ff").arg("rx").arg("2500").arg("94")
CustomBenchmarkBuilder("ff-rx-5000", "", "cacheattack").arg("ff").arg("rx").arg("5000").arg("94")
CustomBenchmarkBuilder("ff-rx-7500", "", "cacheattack").arg("ff").arg("rx").arg("7500").arg("94")
CustomBenchmarkBuilder("ff-rx-10000", "", "cacheattack").arg("ff").arg("rx").arg("10000").arg("94")
CustomBenchmarkBuilder("ff-rx-max", "", "cacheattack").arg("ff").arg("rx").arg("max").arg("94")

CustomBenchmarkBuilder("ru-benign", "", "randomupdates").arg("pp").arg("benign").arg("10000").arg("120")

CustomBenchmarkBuilder("ru-pp-test-2500", "", "randomupdates").arg("pp").arg("test").arg("2500").arg("120")
CustomBenchmarkBuilder("ru-pp-test-5000", "", "randomupdates").arg("pp").arg("test").arg("5000").arg("120")
CustomBenchmarkBuilder("ru-pp-test-7500", "", "randomupdates").arg("pp").arg("test").arg("7500").arg("120")
CustomBenchmarkBuilder("ru-pp-test-10000", "", "randomupdates").arg("pp").arg("test").arg("10000").arg("120")
CustomBenchmarkBuilder("ru-pp-tx-2500", "", "randomupdates").arg("pp").arg("tx").arg("2500").arg("120")
CustomBenchmarkBuilder("ru-pp-tx-5000", "", "randomupdates").arg("pp").arg("tx").arg("5000").arg("120")
CustomBenchmarkBuilder("ru-pp-tx-7500", "", "randomupdates").arg("pp").arg("tx").arg("7500").arg("120")
CustomBenchmarkBuilder("ru-pp-tx-10000", "", "randomupdates").arg("pp").arg("tx").arg("10000").arg("120")
CustomBenchmarkBuilder("ru-pp-rx-2500", "", "randomupdates").arg("pp").arg("rx").arg("2500").arg("120")
CustomBenchmarkBuilder("ru-pp-rx-5000", "", "randomupdates").arg("pp").arg("rx").arg("5000").arg("120")
CustomBenchmarkBuilder("ru-pp-rx-7500", "", "randomupdates").arg("pp").arg("rx").arg("7500").arg("120")
CustomBenchmarkBuilder("ru-pp-rx-10000", "", "randomupdates").arg("pp").arg("rx").arg("10000").arg("120")

CustomBenchmarkBuilder("ru-fr-test-2500", "", "randomupdates").arg("fr").arg("test").arg("2500").arg("120")
CustomBenchmarkBuilder("ru-fr-test-5000", "", "randomupdates").arg("fr").arg("test").arg("5000").arg("120")
CustomBenchmarkBuilder("ru-fr-test-7500", "", "randomupdates").arg("fr").arg("test").arg("7500").arg("120")
CustomBenchmarkBuilder("ru-fr-test-10000", "", "randomupdates").arg("fr").arg("test").arg("10000").arg("120")
CustomBenchmarkBuilder("ru-fr-tx-2500", "", "randomupdates").arg("fr").arg("tx").arg("2500").arg("120")
CustomBenchmarkBuilder("ru-fr-tx-5000", "", "randomupdates").arg("fr").arg("tx").arg("5000").arg("120")
CustomBenchmarkBuilder("ru-fr-tx-7500", "", "randomupdates").arg("fr").arg("tx").arg("7500").arg("120")
CustomBenchmarkBuilder("ru-fr-tx-10000", "", "randomupdates").arg("fr").arg("tx").arg("10000").arg("120")
CustomBenchmarkBuilder("ru-fr-rx-2500", "", "randomupdates").arg("fr").arg("rx").arg("2500").arg("120")
CustomBenchmarkBuilder("ru-fr-rx-5000", "", "randomupdates").arg("fr").arg("rx").arg("5000").arg("120")
CustomBenchmarkBuilder("ru-fr-rx-7500", "", "randomupdates").arg("fr").arg("rx").arg("7500").arg("120")
CustomBenchmarkBuilder("ru-fr-rx-10000", "", "randomupdates").arg("fr").arg("rx").arg("10000").arg("120")

CustomBenchmarkBuilder("ru-ff-test-2500", "", "randomupdates").arg("ff").arg("test").arg("2500").arg("94")
CustomBenchmarkBuilder("ru-ff-test-5000", "", "randomupdates").arg("ff").arg("test").arg("5000").arg("94")
CustomBenchmarkBuilder("ru-ff-test-7500", "", "randomupdates").arg("ff").arg("test").arg("7500").arg("94")
CustomBenchmarkBuilder("ru-ff-test-10000", "", "randomupdates").arg("ff").arg("test").arg("10000").arg("94")
CustomBenchmarkBuilder("ru-ff-tx-2500", "", "randomupdates").arg("ff").arg("tx").arg("2500").arg("94")
CustomBenchmarkBuilder("ru-ff-tx-5000", "", "randomupdates").arg("ff").arg("tx").arg("5000").arg("94")
CustomBenchmarkBuilder("ru-ff-tx-7500", "", "randomupdates").arg("ff").arg("tx").arg("7500").arg("94")
CustomBenchmarkBuilder("ru-ff-tx-10000", "", "randomupdates").arg("ff").arg("tx").arg("10000").arg("94")
CustomBenchmarkBuilder("ru-ff-rx-2500", "", "randomupdates").arg("ff").arg("rx").arg("2500").arg("94")
CustomBenchmarkBuilder("ru-ff-rx-5000", "", "randomupdates").arg("ff").arg("rx").arg("5000").arg("94")
CustomBenchmarkBuilder("ru-ff-rx-7500", "", "randomupdates").arg("ff").arg("rx").arg("7500").arg("94")
CustomBenchmarkBuilder("ru-ff-rx-10000", "", "randomupdates").arg("ff").arg("rx").arg("10000").arg("94")