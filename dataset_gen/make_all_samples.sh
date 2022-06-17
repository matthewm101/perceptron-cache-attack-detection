#!/bin/bash

# This script executes run_simulation.py with all the different benchmarks, one by one.
# It may be possible to rewrite this to run multiple gem5 instances simultaneously,
# but I haven't tested whether gem5 supports that without any issues.

# Make sure to change these if your directory layout differs
cp ./sim_config.py ../gem5/configs/sim_config.py
cp ./run_simulation.py ../gem5/configs/run_simulation.py
cd ../gem5

# Add or remove benchmarks from this array
declare -a benches=(
    # The following benchmarks are useful for testing scenarios where attacker programs try to mimic
    # benign programs when not attacking.
    # I recommend running each of these benchmarks with at least 100 samples, and ru-benign with at least 1000.
    "idletimer"
    "pp-test-2500" "pp-test-5000" "pp-test-7500" "pp-test-10000" 
    "pp-tx-2500" "pp-tx-5000" "pp-tx-7500" "pp-tx-10000" 
    "pp-rx-2500" "pp-rx-5000" "pp-rx-7500" "pp-rx-10000" 
    "fr-test-2500" "fr-test-5000" "fr-test-7500" "fr-test-10000" 
    "fr-tx-2500" "fr-tx-5000" "fr-tx-7500" "fr-tx-10000" 
    "fr-rx-2500" "fr-rx-5000" "fr-rx-7500" "fr-rx-10000" 
    "ff-test-2500" "ff-test-5000" "ff-test-7500" "ff-test-10000" 
    "ff-tx-2500" "ff-tx-5000" "ff-tx-7500" "ff-tx-10000" 
    "ff-rx-2500" "ff-rx-5000" "ff-rx-7500" "ff-rx-10000" 
    "ru-benign"
    "ru-pp-test-2500" "ru-pp-test-5000" "ru-pp-test-7500" "ru-pp-test-10000" 
    "ru-pp-tx-2500" "ru-pp-tx-5000" "ru-pp-tx-7500" "ru-pp-tx-10000" 
    "ru-pp-rx-2500" "ru-pp-rx-5000" "ru-pp-rx-7500" "ru-pp-rx-10000" 
    "ru-fr-test-2500" "ru-fr-test-5000" "ru-fr-test-7500" "ru-fr-test-10000" 
    "ru-fr-tx-2500" "ru-fr-tx-5000" "ru-fr-tx-7500" "ru-fr-tx-10000" 
    "ru-fr-rx-2500" "ru-fr-rx-5000" "ru-fr-rx-7500" "ru-fr-rx-10000" 
    "ru-ff-test-2500" "ru-ff-test-5000" "ru-ff-test-7500" "ru-ff-test-10000" 
    "ru-ff-tx-2500" "ru-ff-tx-5000" "ru-ff-tx-7500" "ru-ff-tx-10000" 
    "ru-ff-rx-2500" "ru-ff-rx-5000" "ru-ff-rx-7500" "ru-ff-rx-10000" 

    # These are SPEC2006 integer-based samples that are useful for making more benign data.
    # These are mostly useful for further testing the specificity of models.
    # However, note that these will be easier to distinguish from the above malicious samples
    # when compared to idletimer and ru-benign.
    
    # "bzip2" "gcc" "mcf" "gobmk" "hmmer" 
    # "sjeng" "libquantum" "h264ref" "omnetpp" "astar"

    # These are malicious samples that run the attack at the max possible speed, offering no time for idling or random updates.
    # These will be easier to identify as malicious than the other malicious samples,
    # but including these in testing may make it more difficult to identify slower samples,
    # assuming that Perspectron's default behavior of rounding datapoint values around 0.5 is used.

    # "pp-test-max"
    # "pp-tx-max"
    # "pp-rx-max"
    # "fr-test-max"
    # "fr-tx-max"
    # "fr-rx-max"
    # "ff-test-max"
    # "ff-tx-max"
    # "ff-rx-max"
)

for i in "${benches[@]}"
do
    ./build/X86/gem5.opt ./configs/run_simulation.py "$i"
done