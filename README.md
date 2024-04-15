# WeShap
Weak Supervision Source Evaluation with Shapley values

## Examples:
Run experiments for ranking LFs on the Youtube dataset
```
python run_pws.py --dataset-name youtube --feature-extractor sentence-bert --lf-tag ngram_random_100 --mode rank
```

Run experiments for correcting LFs on the Indoor-Outdoor dataset
```
python run_pws.py --dataset-name indoor-outdoor --mode revise --revision-type mute
```
