By default, dataset_gen/run_simulation.py will output batches of samples to this folder.
The output folder can be changed by modifying dataset_gen/sim_config.py.

Also by default, dataset_gen/find_data_bounds.py and dataset_gen/generate_dataset.py will read batches from this folder.
So, if you want to create a dataset, ensure that this folder contains everything that you want in the dataset,
since every file with a .pbz2 extension will be read.