## custom-ogbl-split
Create split for link property prediction task for your custom datasets

## Step 1: Run the demo

```bash
python3 pos_neg_split.py --dataset Drug-Drug.edgelist --splitting_strategy random
```

## Step 2: Unpack/Unzip the generated files and get the split_dict.dt for getting the splitted edges

```bash
python get_the_sets.py
```
