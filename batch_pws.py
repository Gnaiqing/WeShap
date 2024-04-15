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

label_models = ["Snorkel", "MV"]
end_models = ["logistic", "knn"]

mode = "revise"  # Mode of the experiment. One of "revise", "rank"
eval_metrics = ["weshap", "acc", "cov", "acc-cov", "if"]  # Metrics used to sort label functions
use_soft_labels = True  # Use soft labels or hard labels
tune_label_model = True  # Tune label model or not
tune_end_model = True  # Tune end model or not
tune_weshap_params = True  # Tune WEShap parameters or not
# distance_metrics = ["cosine", "manhattan"]  # Distance metric for WEShap
# valid_sample_fracs = [50, 100, 200, 400, 800]  # Fraction of validation set used for tuning

abstain_strategy = "discard"  # Strategy for non-covered instances. One of "discard", "keep", "random"
runs = 5  # Number of runs for each experiment
interval = 10  # Interval of runs to save results
metric = "acc"  # Metric used to tune label model and end model

for dataset in datasets:
    for label_model in label_models:
        for end_model in end_models:
            if label_model == "Snorkel" and end_model == "logistic":
                continue
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

            for eval_metric in eval_metrics:
                cmd = f"python run_pws.py --dataset-name {dataset} --feature-extractor {feature_extractor} " \
                      f"--mode {mode} --revision-type mute --abstain-strategy {abstain_strategy} " \
                      f"--tune-metric {metric} --label-model {label_model} " \
                      f"--end-model {end_model} --eval-metric {eval_metric} --save-wandb --runs {runs} --interval {interval}"

                if use_soft_labels:
                    cmd += " --use-soft-labels"

                if lf_tag is not None:
                    cmd += f" --lf-tag {lf_tag}"

                if tune_label_model:
                    cmd += " --tune-label-model"

                if tune_end_model:
                    cmd += " --tune-end-model"

                if tune_weshap_params:
                    cmd += " --tune-weshap-params"

                if eval_metric == "if":
                    cmd += " --load-if --save-if"

                # if eval_metric == "weshap":
                #     for valid_sample_frac in valid_sample_fracs:
                #         cur_cmd = cmd + f" --valid-sample-frac {valid_sample_frac}"
                #         print(cur_cmd)
                #         os.system(cur_cmd)

                print(cmd)
                os.system(cmd)
