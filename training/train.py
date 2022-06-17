import argparse
import configparser
from itertools import chain
import pandas as pd
import numpy as np
import random
import models

ap = argparse.ArgumentParser()
ap.add_argument("config", type=argparse.FileType("r"))
args = ap.parse_args()
config = configparser.ConfigParser(allow_no_value=True)
config.optionxform=str
config.read_file(args.config)

# The proportion of each source's samples to use for training. The remainder will be used for testing.
source_train_props = {}
# The amount of weight given to each source during training. Weight affects the probability that a sample from a specific source will be drawn when a malicious or benign sample is needed.
source_train_weights = {}
# The amount of weight given to each source during testing. This only affects the aggregate accuracy score shown at the end of testing.
source_test_weights = {}
# The names of the sources labeled "benign" in the configuration file.
benign_sources = []
# The names of the sources labeled "malicious" in the configuration file.
malicious_sources = []
# A mapping from each source name being used to a list of samples for that source. 
samples = {}

# Each source listed under the [Source Settings] section of the configuration file is parsed here.
for k,v in config["Source Settings"].items():
    splits = v.split(",")
    if splits[0].strip().lower() in ["false", "0", "good", "benign"]:
        benign_sources.append(k.strip())
    elif splits[0].strip().lower() in ["true", "1", "bad", "malicious"]:
        malicious_sources.append(k.strip())
    else:
        print(f"Bad line under Source Settings: {k} = {v}")
        exit(0)
    samples[k.strip()] = []
    source_train_props[k.strip()] = float(splits[1].strip())
    source_train_weights[k.strip()] = int(splits[2].strip())
    source_test_weights[k.strip()] = int(splits[3].strip())

# The number of samples used during each epoch.
samples_per_epoch = int(config["Experiment Settings"]["samples_per_epoch"])
# The max number of epochs run during training.
max_epochs = int(config["Experiment Settings"]["max_epochs"])
# The number of malicious samples to train with in each epoch. For example, setting this to 500 with a samples_per_epoch of 1000 will make half of the samples be malicious, and the other half benign.
# This can either be a single integer or a list of integers. If multiple integers are used, then each value will be testing with in different runs.
bad_samples_per_epochs = list(map(lambda x: int(x.strip()), config["Experiment Settings"]["malicious_samples_per_epoch"].split(",")))
for bspe in bad_samples_per_epochs:
    assert bspe >= 0 and bspe <= samples_per_epoch
# The required accuracy for training to end. 
stop_acc =  float(config["Experiment Settings"]["early_stopping_accuracy"])
# The seed or list of seeds to use for RNG. If multiple seeds are specified, a run will be performed with each seed.
rng_seeds = list(map(lambda x: int(x.strip()), config["Experiment Settings"]["rng_seed"].split(",")))
# The name of the model to use. To make a new model, add it to models.py and register its name in the model_map.
model_name = config["Experiment Settings"]["model"]
# How many contiguous samples to concat together to form the samples actually used for testing.
# This provides a simple way to make samples that capture longer periods of time.
concat_amount = int(config["Experiment Settings"]["sample_concat_amount"])
assert concat_amount >= 1


# The raw pandas dataframe containing the samples.
dataset = pd.read_csv(config["File Settings"]["dataset"])

# The list of datapoints to use exclusively (ignoring all other datapoints for training and testing). If this list is empty, all datapoints are considered.
whitelist = list(config["Datapoint Whitelist"].keys())
# The list of datapoint to ignore for training and testing. If a datapoint is on both the whitelist and blacklist, the blacklist takes priority and the datapoint is ignored.
blacklist = list(config["Datapoint Blacklist"].keys())

# The list of datapoint names in the order in which they will be used.
sample_order = []

# This iterates through all the available datapoints and keeps only the ones allowed by the whitelist/blacklist.
for full_c in dataset.columns:
    if str(full_c).startswith("in/"):
        c = full_c[3:]
        for bl in blacklist:
            if c.startswith(bl):
                break
        else:
            found = False
            for wl in whitelist:
                if c.startswith(wl):
                    found = True
                    break
            if found or len(whitelist) == 0:
                sample_order.append(full_c)

n_datapoints = len(sample_order)

print("Loading dataset...",end="",flush=True)
# This connects source names to dictionaries, which themselves connect sample numbers to the row's data.
raw_rows = {k:{} for k in samples.keys()}
for i, row in dataset.iterrows():
    if row["source"] in samples.keys():
        raw_rows[row["source"]][row["sample_number"]] = np.array([row[x] for x in sample_order])
# This step combines multiple contiguous samples together, if specified using sample_concat_amount. Otherwise it just adds the samples to the samples map.
for src, rowmap in raw_rows.items():
    for n, row in rowmap.items():
        s = row.copy()
        if concat_amount > 1:
            broken = False
            for i in range(1, concat_amount):
                if (n+i) in rowmap.keys():
                    s = np.concatenate((s, rowmap[n+i]))
                else:
                    broken = True
                    break
            if broken:
                continue
        samples[src].append(s)
if concat_amount > 1:
    sample_order = np.tile(sample_order, concat_amount)
    n_datapoints = len(sample_order)
print("done.")

# This class encapsulates the process of randomly sampling samples from all the different sources.
# To ensure that each source is used equally, source names drawn from a bucket each time a specific sample type (good or bad) is needed.
# When the bucket empties, each source is added back to the bucket as many times as its training weight.
# Similarly, to ensure that each sample from each source is used equally, buckets are filled with samples from the training slice.
# The separation of each source's samples into the training and testing slices and the actual random draws are seeded by a given number.
class Sampler:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.sample_train_indices = {}
        self.sample_test_indices = {}
        for src, samps in samples.items():
            n_samps = len(samps)
            n_train_samps = int(n_samps * source_train_props[src])
            samp_inds = list(range(n_samps))
            self.rng.shuffle(samp_inds)
            train_samps, test_samps = samp_inds[:n_train_samps], samp_inds[n_train_samps:]
            if len(train_samps) > 0:
                self.sample_train_indices[src] = train_samps
            if len(test_samps) > 0:
                self.sample_test_indices[src] = test_samps
        self.good_bucket = []
        self.bad_bucket = []
        self.sample_buckets = {c:[] for c in samples.keys()}
    def pop_good_bucket(self):
        if len(self.good_bucket) == 0:
            self.good_bucket = list(chain(*[[k] * source_train_weights[k] for k in self.sample_train_indices.keys() if (k in benign_sources)]))
        rand_i = self.rng.randint(0, len(self.good_bucket)-1)
        cl = self.good_bucket[rand_i]
        if rand_i == len(self.good_bucket) - 1:
            self.good_bucket.pop()
        else:
            self.good_bucket[rand_i] = self.good_bucket.pop()
        return cl
    def pop_bad_bucket(self):
        if len(self.bad_bucket) == 0:
            self.bad_bucket = list(chain(*[[k] * source_train_weights[k] for k in self.sample_train_indices.keys() if (k in malicious_sources)]))
        rand_i = self.rng.randint(0, len(self.bad_bucket)-1)
        cl = self.bad_bucket[rand_i]
        if rand_i == len(self.bad_bucket) - 1:
            self.bad_bucket.pop()
        else:
            self.bad_bucket[rand_i] = self.bad_bucket.pop()
        return cl
    def pop_sample_bucket(self, src):
        if len(self.sample_buckets[src]) == 0:
            self.sample_buckets[src] = self.sample_train_indices[src].copy()
        rand_i = self.rng.randint(0, len(self.sample_buckets[src])-1)
        samp_i = self.sample_buckets[src][rand_i]
        if rand_i == len(self.sample_buckets[src]) - 1:
            self.sample_buckets[src].pop()
        else:
            self.sample_buckets[src][rand_i] = self.sample_buckets[src].pop()
        return samples[src][samp_i]
    def draw_good_train(self):
        src = self.pop_good_bucket()
        return self.pop_sample_bucket(src)
    def draw_bad_train(self):
        src = self.pop_bad_bucket()
        return self.pop_sample_bucket(src)
    def get_all_train_samples(self, src):
        return [samples[src][i] for i in self.sample_train_indices[src]]
    def get_all_test_samples(self, src):
        return [samples[src][i] for i in self.sample_test_indices[src]]
        
# This method runs an experiment with a given number of bad samples per epoch and a given randomness seed.
def run_experiment(bad_per_epoch, seed, verbose=True):
    print(f"Simulating with seed {seed} and {bad_per_epoch} bad samples/epoch ({bad_per_epoch / samples_per_epoch * 100}%).")
    sampler = Sampler(seed)
    model: models.DetectorModel = models.model_map[model_name](n_datapoints)
    good_per_epoch = samples_per_epoch - bad_per_epoch
    for epoch in range(max_epochs):
        # During each epoch, exactly bad_per_epoch malicious samples are drawn, with the remaining being good samples.
        # The order in which samples from each class are drawn is completely random.
        sample_classes = [False for _ in range(good_per_epoch)] + [True for _ in range(bad_per_epoch)]
        sampler.rng.shuffle(sample_classes)
        # True/false positives/negatives are recorded across all training samples.
        tns, fps, tps, fns = 0, 0, 0, 0
        for c in sample_classes:
            if c:
                if model.train(sampler.draw_bad_train(), True):
                    tps += 1
                else:
                    fns += 1
            else:
                if model.train(sampler.draw_good_train(), False):
                    tns += 1
                else:
                    fps += 1
        specificity = tns / good_per_epoch
        sensitivity = tps / bad_per_epoch
        accuracy = (tps + tns) / samples_per_epoch
        if verbose:
            print(f"Epoch {epoch+1}: sensitivity = {sensitivity}, specificity = {specificity}, accuracy = {accuracy}")
        if accuracy >= stop_acc:
            if verbose:
                print(f"The accuracy reached {stop_acc}; training is over.")
            break
    print(f"Training finished in {epoch+1} epochs.")

    # This prints out the 5 datapoints that are assigned the highest and lowest weights.
    # Note that this code is hard-coded for the default perceptron model, and should be removed if testing with other models.
    if verbose:
        model_weights = [(model.weights[i],i) for i in range(len(model.weights))]
        model_weights.sort()
        print("Top five datapoints for malicious classification:")
        for w,i in reversed(model_weights[-5:]):
            print(f"  {sample_order[i]}[{w}]")
        print("Top five datapoints for benign classification:")
        for w,i in model_weights[:5]:
            print(f"  {sample_order[i]}[{w}]")

    # Given a list of samples and the type of output expected from all the samples, this returns the numbers of correct and incorrect classifications.
    def test_with_samples(samps, output):
        trues = 0
        falses = 0
        for samp in samps:
            if model.test(samp, output):
                trues += 1
            else:
                falses += 1
        return trues, falses

    # These next three blocks of code go through all of the training and testing splits for each source of samples and compute the training/testing accuracy for each source.
    # These accuracies are then combined together (according to the weights specified in the config file) to generate an aggregate training/testing accuracy for all benign or all malicious sources.
    if verbose:
        print("Testing with benign sources...")
    train_tns = 0
    train_btotal = 0
    test_tns = 0
    test_btotal = 0
    for src in benign_sources:
        train_samps = sampler.get_all_train_samples(src)
        test_samps = sampler.get_all_test_samples(src)
        if len(train_samps) == 0 and len(test_samps) == 0:
            continue
        if verbose:
            print(f"\t{src}: ", end="")
        if len(train_samps) > 0:
            tns, fps = test_with_samples(train_samps, False)
            train_tns += (tns / (tns + fps)) * source_train_weights[src]
            train_btotal += source_train_weights[src]
            if verbose:
                print(f"TrainAcc = {tns / (tns + fps)}", end="")
        if verbose and len(train_samps) > 0 and len(test_samps) > 0:
            print(", ", end="")
        if len(test_samps) > 0:
            tns, fps = test_with_samples(test_samps, False)
            test_tns += (tns / (tns + fps)) * source_test_weights[src]
            test_btotal += source_test_weights[src]
            if verbose:
                print(f"TestAcc = {tns / (tns + fps)}", end="")
        if verbose:
            print("")

    if verbose:
        print("Testing with malicious sources...")
    train_tps = 0
    train_mtotal = 0
    test_tps = 0
    test_mtotal = 0
    for src in malicious_sources:
        train_samps = sampler.get_all_train_samples(src)
        test_samps = sampler.get_all_test_samples(src)
        if len(train_samps) == 0 and len(test_samps) == 0:
            continue
        if verbose:
            print(f"\t{src}: ", end="")
        if len(train_samps) > 0:
            tps, fns = test_with_samples(train_samps, True)
            train_tps += (tps / (tps + fns)) * source_train_weights[src]
            train_mtotal += source_train_weights[src]
            if verbose:
                print(f"TrainAcc = {tps / (tps + fns)}", end="")
        if verbose and len(train_samps) > 0 and len(test_samps) > 0:
            print(", ", end="")
        if len(test_samps) > 0:
            tps, fns = test_with_samples(test_samps, True)
            test_tps += (tps / (tps + fns)) * source_test_weights[src]
            test_mtotal += source_test_weights[src]
            if verbose:
                print(f"TestAcc = {tps / (tps + fns)}", end="")
        if verbose:
            print("")
    
    train_spec = train_tns / train_btotal
    test_spec = test_tns / test_btotal
    train_sen = train_tps / train_mtotal
    test_sen = test_tps / test_mtotal
    print(f"Sensitivity: {train_sen} train, {test_sen} test")
    print(f"Specificity: {train_spec} train, {test_spec} test")
    return test_sen, test_spec


# This last block contains the code that handles several different ways the config file can be set up. Particularly:
#   If one seed and one bad_samples_per_epoch value are specified, then the one experiment is run in verbose mode, printing out the weights of the model and the accuracies for each sample.
#   If multiple seeds are specified with one bad_samples_per_epoch value, then verbose mode is disabled and aggregate states across all the seeds are printed.
#   If one seed and multiple bad_samples_per_epoch values are specified, then non-verbose experiments are run for each different BSpE value with the same seed. The best BSpE value is printed.
#   If multiple seeds and multiple BSpE values are specified, then non-verbose experiments are run for each combination of BSpE value and seed.
#       The average across all seeds for each BSpE value is printed, along with aggregate stats that consider both individual seed-BSpE combos and the all-seed averages for each BSpE value.

if len(bad_samples_per_epochs) == 1:
    if len(rng_seeds) == 1:
        run_experiment(bad_samples_per_epochs[0], rng_seeds[0], True)
    else:
        best_sen = (0, "N/A")
        worst_sen = (1, "N/A")
        best_spec = (0, "N/A")
        worst_spec = (1, "N/A")
        best_sen_with_perfect_spec = (0, "N/A")
        best_spec_with_perfect_sen = (0, "N/A")
        best_sum = (0, "N/A")
        avg_sen = 0
        avg_spec = 0
        for s in rng_seeds:
            sen, spec = run_experiment(bad_samples_per_epochs[0], s, False)
            print()
            if sen > best_sen[0]:
                best_sen = (sen, s)
            if sen < worst_sen[0]:
                worst_sen = (sen, s)
            if spec > best_spec[0]:
                best_spec = (spec, s)
            if spec < worst_spec[0]:
                worst_spec = (spec, s)
            if sen == 1 and spec > best_spec_with_perfect_sen[0]:
                best_spec_with_perfect_sen = (spec, s)
            if spec == 1 and sen > best_sen_with_perfect_spec[0]:
                best_sen_with_perfect_spec = (sen, s)
            if sen + spec > best_sum[0]:
                best_sum = (sen + spec, s)
            avg_sen += sen
            avg_spec += spec
        avg_sen /= len(rng_seeds)
        avg_spec /= len(rng_seeds)
        print(f"The highest sensitivity achieved was {best_sen[0]} with seed {best_sen[1]}.")
        print(f"The lowest sensitivity achieved was {worst_sen[0]} with seed {worst_sen[1]}.")
        print(f"The highest specificity achieved was {best_spec[0]} with seed {best_spec[1]}.")
        print(f"The lowest specificity achieved was {worst_spec[0]} with seed {worst_spec[1]}.")
        print(f"The highest sensitivity achieved while maintaining perfect specificity was {best_sen_with_perfect_spec[0]} with seed {best_sen_with_perfect_spec[1]}.")
        print(f"The highest specificity achieved while maintaining perfect sensitivity was {best_spec_with_perfect_sen[0]} with seed {best_spec_with_perfect_sen[1]}.")
        print(f"The highest sum of sensitivity and specificity achieved was {best_sum[0]} with seed {best_sum[1]}.")
        print()
        print(f"Average testing sensitivity across all seeds: {avg_sen}")
        print(f"Average testing specificity across all seeds: {avg_spec}")
else:
    if len(rng_seeds) == 1:
        best_spec_with_perfect_sen = (0, "N/A")
        best_sen_with_perfect_spec = (0, "N/A")
        best_sum = (0, "N/A")
        for bspe in bad_samples_per_epochs:
            sen, spec = run_experiment(bspe, rng_seeds[0], False)
            print()
            if sen == 1 and spec > best_spec_with_perfect_sen[0]:
                best_spec_with_perfect_sen = (spec, f"{bspe} malicious samples per epoch")
            if spec == 1 and sen > best_sen_with_perfect_spec[0]:
                best_sen_with_perfect_spec = (sen, f"{bspe} malicious samples per epoch")
            if sen + spec > best_sum[0]:
                best_sum = (sen + spec, f"{bspe} malicious samples per epoch")
        print(f"The best specificity achieved while maintaining perfect sensitivity was {best_spec_with_perfect_sen[0]} with {best_spec_with_perfect_sen[1]}.")
        print(f"The best sensitivity achieved while maintaining perfect specificity was {best_sen_with_perfect_spec[0]} with {best_sen_with_perfect_spec[1]}.")
        print(f"The best sum of sensitivity and specificity achieved was {best_sum[0]} with {best_sum[1]}.")
    else:
        NEAR_PERFECT_THRESHOLD_PERCENT = 95

        ind_best_spec_with_perfect_sen = (0, "N/A")
        ind_best_sen_with_perfect_spec = (0, "N/A")
        ind_best_sum = (0, "N/A")
        avg_best_spec_with_near_perfect_sen = (0, "N/A")
        avg_best_sen_with_near_perfect_spec = (0, "N/A")
        avg_best_sum = (0, "N/A")
        for bspe in bad_samples_per_epochs:
            avg_sen = 0
            avg_spec = 0
            for s in rng_seeds:
                sen, spec = run_experiment(bspe, s, False)
                if sen == 1 and spec > ind_best_spec_with_perfect_sen[0]:
                    ind_best_spec_with_perfect_sen = (spec, f"{bspe} malicious samples per epoch, seed {s}")
                if spec == 1 and sen > ind_best_sen_with_perfect_spec[0]:
                    ind_best_sen_with_perfect_spec = (sen, f"{bspe} malicious samples per epoch, seed {s}")
                if sen + spec > ind_best_sum[0]:
                    ind_best_sum = (sen + spec, f"{bspe} malicious samples per epoch, seed {s}")
                avg_sen += sen
                avg_spec += spec
            avg_sen /= len(rng_seeds)
            avg_spec /= len(rng_seeds)
            if avg_sen >= (NEAR_PERFECT_THRESHOLD_PERCENT / 100) and avg_spec > avg_best_spec_with_near_perfect_sen[0]:
                avg_best_spec_with_near_perfect_sen = (avg_spec, f"{bspe} malicious samples per epoch")
            if avg_spec >= (NEAR_PERFECT_THRESHOLD_PERCENT / 100) and avg_sen > avg_best_sen_with_near_perfect_spec[0]:
                avg_best_sen_with_near_perfect_spec = (avg_sen, f"{bspe} malicious samples per epoch")
            if avg_sen + avg_spec > avg_best_sum[0]:
                avg_best_sum = (avg_sen + avg_spec, f"{bspe} malicious samples per epoch")
            print()
            print(f"Average sensitivity with {bspe} bad samples/epoch: {avg_sen}")
            print(f"Average specificity with {bspe} bad samples/epoch: {avg_spec}")
            print()
        print("When considering only individual seeds:")
        print(f"\tThe best specificity achieved while maintaining perfect sensitivity was {ind_best_spec_with_perfect_sen[0]} with {ind_best_spec_with_perfect_sen[1]}.")
        print(f"\tThe best sensitivity achieved while maintaining perfect specificity was {ind_best_sen_with_perfect_spec[0]} with {ind_best_sen_with_perfect_spec[1]}.")
        print(f"\tThe best sum of sensitivity and specificity achieved was {ind_best_sum[0]} with {ind_best_sum[1]}.")
        print("When considering averages across all seeds:")
        print(f"\tThe best specificity achieved while maintaining near-perfect (>{NEAR_PERFECT_THRESHOLD_PERCENT}%) sensitivity was {avg_best_spec_with_near_perfect_sen[0]} with {avg_best_spec_with_near_perfect_sen[1]}.")
        print(f"\tThe best sensitivity achieved while maintaining near-perfect (>{NEAR_PERFECT_THRESHOLD_PERCENT}%) specificity was {avg_best_sen_with_near_perfect_spec[0]} with {avg_best_sen_with_near_perfect_spec[1]}.")
        print(f"\tThe best sum of sensitivity and specificity achieved was {avg_best_sum[0]} with {avg_best_sum[1]}.")
            