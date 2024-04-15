import os
datasets = [
    "youtube",
    "imdb",
    "yelp",
    "trec",
    "medical_abstract",
    "mushroom",
    "census",
    "KSDD-1",
    "indoor-outdoor",
    "voc07-animal",
]

numeric_datasets = ("census", "mushroom", "KSDD-1", "indoor-outdoor", "voc07-animal")
multiclass_datasets = ("trec", "medical_abstract")

label_models = ["Snorkel"]
end_models = ["logistic"]

mode = "denoise"  # Mode of the experiment. One of "denoise", "time"

for dataset in datasets:
    for label_model in label_models:
        for end_model in end_models:
            if dataset in numeric_datasets:
                if dataset == "mushroom":
                    lf_tag = "feature_exhaustive"
                else:
                    lf_tag = None
                feature_extractor = "None"
            else:
                lf_tag = "ngram_random_100"
                if dataset in ["imdb", "yelp"]:
                    feature_extractor = "bertweet-sentiment"
                else:
                    feature_extractor = "sentence-bert"

            cmd = f"python revise.py --dataset-name {dataset} --feature-extractor {feature_extractor} " \
                  f" --mode {mode} --label-model {label_model} --end-model {end_model} " \
                  f"--load-if --use-soft-labels --tune-label-model --save-wandb --weshap-k 40"

            if lf_tag is not None:
                cmd += f" --lf-tag {lf_tag}"

            print(cmd)
            os.system(cmd)
