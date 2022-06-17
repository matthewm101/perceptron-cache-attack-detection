# Copyright (c) 2012-2013 ARM Limited
# All rights reserved.
#
# The license below extends only to copyright in the software and shall
# not be construed as granting a license to any other intellectual
# property including but not limited to intellectual property relating
# to a hardware implementation of the functionality of the software
# licensed hereunder.  You may use the software subject to the license
# terms below provided that you ensure that this notice is replicated
# unmodified and in its entirety in all distributions of the software,
# modified or unmodified, in source code or in binary form.
#
# Copyright (c) 2006-2008 The Regents of The University of Michigan
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Authors: Steve Reinhardt

# This script is a modification of gem5/configs/example/se.py.

import multiprocessing
import pickle
import bz2
from gem5.simulate.exit_event import ExitEvent
from m5.stats.gem5stats import get_simstat

import sim_config

import sys

import m5
from m5.defines import buildEnv
from m5.objects import *
from m5.util import fatal
from m5.params import Tick

from common import Simulation
from common import CacheConfig
from common import ObjectList
from common import MemConfig
from common.Caches import *
from common.cpu2000 import *

def gen_samples_from_benchmark(bench_name: str):

    # Benchmarks should be defined in sim_config.py before being passed as an argument.
    if bench_name not in sim_config.benchmap.keys():
        print(f"Unrecognized SPEC2006 benchmark provided: {bench_name}. Exiting.")
        sys.exit(1)
    else:
        process = sim_config.benchmap[bench_name]

    # The CPU type is pulled from the config.
    CPUClass, test_mem_mode = Simulation.getCPUClass(sim_config.options.cpu_type)
    CPUClass.numThreads = 1

    np = 1  # Currently fixed to one processor
    system = System(cpu = [CPUClass(cpu_id=i) for i in range(np)],
                    mem_mode = test_mem_mode,
                    mem_ranges = [AddrRange(sim_config.options.mem_size)],
                    cache_line_size = sim_config.options.cacheline_size)

    # Setting up the system
    system.voltage_domain = VoltageDomain(voltage = sim_config.options.sys_voltage)
    system.clk_domain = SrcClockDomain(clock =  sim_config.options.sys_clock, voltage_domain = system.voltage_domain)
    system.cpu_voltage_domain = VoltageDomain()
    system.cpu_clk_domain = SrcClockDomain(clock = sim_config.options.cpu_clock, voltage_domain = system.cpu_voltage_domain)
    for cpu in system.cpu:
        cpu.clk_domain = system.cpu_clk_domain

    # I experimented with using kvm to speed past the first 0.01 to 0.1 seconds of simulation time, but had some issues getting it to work.
    # So, this block will almost never do anything.
    if ObjectList.is_kvm_cpu(CPUClass):
        if buildEnv['TARGET_ISA'] == 'x86':
            system.kvm_vm = KvmVM()
            process.useArchPT = True
            process.kvmInSE = True
        else:
            fatal("KvmCPU can only be used in SE mode with x86")

    # The process selected from the config is loaded onto the CPU.
    for i in range(np):
        system.cpu[i].workload = process       
        system.cpu[i].createThreads()
    
    # Setting up the memory with the specs defined in sim_config.py
    MemClass = Simulation.setMemClass(sim_config.options)
    system.membus = SystemXBar()
    system.system_port = system.membus.cpu_side_ports
    CacheConfig.config_cache(sim_config.options, system)
    MemConfig.config_mem(sim_config.options, system)

    # Finishing the system setup
    system.workload = SEWorkload.init_compatible(process.executable)
    root = Root(full_system = False, system = system)

    # The following code is copied from Simulation.run()
    # Some unneeded options were removed, and modifications made to allow periodic stat dumps

   # Disable some default options
    from m5 import options
    options.dump_config = None
    options.json_config = None
    options.dot_config = None

    root.apply_config(sim_config.options.param)
    m5.instantiate()    # Setup is complete, now the simulations can occur

    # skipped_time is the initial amount of time that is simulated without the debug counters being dumped.
    skipped_time = m5.ticks.fromSeconds(float(sim_config.SKIPPED_TIME))

    print(f"Starting the simulation for {bench_name}")
    print(f"Simulating the first {sim_config.SKIPPED_TIME} second(s) without recording stats...")
    exit_event = m5.simulate(skipped_time)
    print(f"Initial simulation finished.")
    m5.stats.reset()
    
    # Lists of samples are maintained for each sample length.
    samples = {}
    for period in sim_config.SAMPLING_PERIODS:
        samples[period] = []
    
    # To make things run a bit faster, subprocesses are spawned to save each batch of samples.
    # Each batch is a pickle dump of a list of samples, where each sample is a dictionary.
    # Batches are compressed with bz2 to save space.
    file_threads = []
    def save(first, last, samples):
        for period in sim_config.SAMPLING_PERIODS:
            series_name = f"{bench_name}_{first}-{last}_{int(period * 1e6)}us"
            outfile_name = f"{sim_config.OUTPUT_DIR}/{series_name}.pbz2"
            print(f"Saving data to {outfile_name}...")
            with bz2.BZ2File(outfile_name,"wb") as f:
                pickle.dump(samples[period],f)

    broken = False
    for sample in range(sim_config.SAMPLES_PER_BATCH * sim_config.BATCHES_PER_BENCHMARK):
        # See SAMPLING_PERIODS in sim_config.py for a description of how this works.
        # In short, the largest listed sampling period has its samples occur one after another in simulated time,
        # and shorter sampling periods have their samples formed by "trimming" the larger samples (by dumping counters early).
        for p in range(len(sim_config.SAMPLING_PERIODS)):
            print(f"Generating sample {sample} from {sim_config.SAMPLING_PERIODS[p]} second(s) of simulated time...")
            if p == 0:
                sampling_time = m5.ticks.fromSeconds(float(sim_config.SAMPLING_PERIODS[0]))
            else:
                sampling_time = m5.ticks.fromSeconds(float(sim_config.SAMPLING_PERIODS[p]-sim_config.SAMPLING_PERIODS[p-1]))
            exit_event = m5.simulate(sampling_time)
            if ExitEvent.translate_exit_status(exit_event.getCause()) == ExitEvent.MAX_TICK:
                samples[sim_config.SAMPLING_PERIODS[p]].append(get_simstat(root).to_json())
            else:
                print(f"Simulation ended for unexpected reason: {exit_event}")
                broken = True
                break
        # Keyboard interrupts or other sudden simulation ends will immediately break the loop,
        # and any samples in the current batch will be lost.
        if broken:
            break
        m5.stats.reset()
        # Upon recording the last sample in a batch, the batch is saved to a file.
        if sample % sim_config.SAMPLES_PER_BATCH == sim_config.SAMPLES_PER_BATCH - 1:
            file_threads.append(
                multiprocessing.Process(
                    target = save,
                    args=(
                        sample - sim_config.SAMPLES_PER_BATCH + 1,
                        sample,
                        samples
                    )
                )
            )
            file_threads[-1].start()
            samples = {}
            for period in sim_config.SAMPLING_PERIODS:
                samples[period] = []
    print(f"Finished generating samples from benchmark {bench_name}.")    
    print("Joining file saving threads...")
    for t in file_threads:
        t.join()
    print("All samples saved.")

# Currently, only one argument is read: the benchmark to run (as defined in sim_config.py).
# You may want to add another argument to control BATCHES_PER_BENCHMARK or any of the other config options.
gen_samples_from_benchmark(sys.argv[1])