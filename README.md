### PyTorch implementation of Walking Memory

### Dependencies
* PyTorch >= 0.2.0
* python == 2.7 or python == 3.6

## Training
I have set up training with most default params on a very small dataset so that it is easier to get started. Just running the script should work.
```
/bin/bash run.sh ./config.sh
```
### Data
The processed data (train/dev/test split) is stored in data_formatted/ directory.
To download the KB files used for the project run,
```
sh get_data.sh
```
After downloading the data, you will have to change the appropriate entries in the config.sh file (kb_file and text_kb_file).

### Pretrained embeddings
The embeddings used for initializing the network can be downloaded from [here](http://iesl.cs.umass.edu/downloads/spades/entity_lookup_table_50.pkl.gz)

### Model outputs
We are also releasing the output predictions of our model for comparison. Find them in the model_outputs directory.

