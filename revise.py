"""
Run the PWS pipeline
"""
import argparse
import time

from data_utils import load_wrench_data
from utils import print_dataset_stats
import numpy as np
import os
from utils import run_pws_pipeline, get_lf_orders, load_lf_names,  compute_if_score, plot_correction, get_wrench_label_model
import wandb
from pathlib import Path
from snorkel.labeling.analysis import LFAnalysis
from weshap import WeShapAnalysis
import warnings

warnings.filterwarnings("ignore")

dataset_name_map = {
    "youtube": "YouTube",
    "imdb": "IMDB",
    "yelp": "Yelp",
    "medical_abstract": "MedAbs",
    "trec": "TREC",
    "mushroom": "Mushroom",
    "census": "Census",
    "KSDD-1": "KSDD",
    "indoor-outdoor": "Indoor-Outdoor",
    "voc07-animal": "VOC07-Animal",
}


def main(args):
    train_dataset, valid_dataset, test_dataset = load_wrench_data(data_root=args.dataset_path,
                                                                  dataset_name=args.dataset_name,
                                                                  feature=args.feature_extractor)

    print(f"Dataset path: {args.dataset_path}, name: {args.dataset_name}")
    print_dataset_stats(train_dataset, split="train")
    print_dataset_stats(valid_dataset, split="valid")
    print_dataset_stats(test_dataset, split="test")
    if args.save_wandb:
        group_id = wandb.util.generate_id()
        config_dict = vars(args)
        config_dict["group_id"] = group_id

    if args.lf_tag is not None:
        # load LF from file
        lf_path = f"{args.dataset_path}/{args.dataset_name}/lf_{args.lf_tag}.npz"
        if not os.path.exists(lf_path):
            print(f"LF file {lf_path} does not exist")
            exit(1)
        wl = np.load(lf_path)
        train_dataset.weak_labels = wl["L_train"].tolist()
        valid_dataset.weak_labels = wl["L_valid"].tolist()
        train_dataset.n_lf = len(train_dataset.weak_labels[0])
        valid_dataset.n_lf = len(valid_dataset.weak_labels[0])

    L_train = np.array(train_dataset.weak_labels)
    L_valid = np.array(valid_dataset.weak_labels)
    summary = LFAnalysis(L_train).lf_summary(Y=np.array(train_dataset.labels))
    lf_names = load_lf_names(args)
    if lf_names is not None:
        summary["lf_names"] = lf_names

    pws_configs = {
        "label_model": args.label_model,
        "end_model": args.end_model,
        "tune_label_model": args.tune_label_model,
        "tune_end_model": args.tune_end_model,
        "tune_metric": args.tune_metric,
        "abstain_strategy": args.abstain_strategy,
        "distance_metric": args.distance_metric,
        "use_soft_labels": args.use_soft_labels
    }
    if args.mode == "denoise":
        train_dataset = train_dataset.get_covered_subset()
        _, _, _, label_model, _ = run_pws_pipeline(train_dataset, valid_dataset, test_dataset,
                                                   **pws_configs, return_models=True)
        y_pred = label_model.predict(train_dataset)
        analysis = WeShapAnalysis(train_dataset, valid_dataset,
                                  n_neighbors=args.weshap_k,
                                  weights=args.weshap_weights,
                                  metric=args.weshap_distance_metric)
        weshap_contributions = analysis.calculate_contribution().sum(axis=-1)
        y_train = np.array(train_dataset.labels)
        error = (y_train != y_pred).astype(int)
        total_error = np.sum(error)
        print(f"Total error: {total_error}")
        print(f"Error rate: {total_error / len(y_train)}")
        weshap_sorted_error = np.cumsum(error[np.argsort(weshap_contributions)])

        # try correcting the error using if score
        if_path = Path(
            args.dataset_path) / args.dataset_name / f"if_{args.if_type}_{args.if_mode}_{args.label_model}.npy"
        if_score = compute_if_score(train_dataset, valid_dataset,
                                    if_type=args.if_type, mode=args.if_mode, label_model=args.label_model,
                                    load_if=args.load_if, save_if=args.save_if, if_path=if_path)

        if_score = np.sum(if_score, axis=(1,2))
        if_sorted_error = np.cumsum(error[np.argsort(if_score)])
        # plot the error rate
        title = dataset_name_map[args.dataset_name]
        save_path = Path(args.dataset_path) / args.dataset_name / f"{args.dataset_name}_{args.label_model}_denoise-auc-{args.weshap_k}.png"
        plot_correction(weshap_sorted_error, if_sorted_error, total_error, title=title, save_path=save_path)

    elif args.mode == "time":
        for run in range(args.runs):
            if args.save_wandb:
                wandb.init(
                    project="LLMDP-eval",
                    config=config_dict
                )
            st1 = time.process_time()
            summary = LFAnalysis(L_train).lf_summary(Y=np.array(train_dataset.labels))
            accuracy = summary["Emp. Acc."]
            st2 = time.process_time()
            print(f"Time for Accuracy score: {st2-st1:.2f} s")
            summary = LFAnalysis(L_train).lf_summary(Y=np.array(train_dataset.labels))
            coverage = summary["Coverage"]
            st3 = time.process_time()
            print(f"Time for Coverage score: {st3-st2:.2f} s")
            summary = LFAnalysis(L_train).lf_summary(Y=np.array(train_dataset.labels))
            iws_score = (2*summary["Emp. Acc."] - 1) * summary["Coverage"]
            st4 = time.process_time()
            print(f"Time for IWS score: {st4-st3:.2f} s")
            analysis = WeShapAnalysis(train_dataset, valid_dataset,
                                      n_neighbors=args.weshap_k,
                                      weights=args.weshap_weights,
                                      metric=args.weshap_distance_metric)
            weshap_contributions = analysis.calculate_contribution()
            st5 = time.process_time()
            print(f"Time for WeShap: {st5-st4:.2f} s")
            if_score = compute_if_score(train_dataset, valid_dataset,
                                        if_type=args.if_type, mode=args.if_mode, label_model=args.label_model,
                                        load_if=False, save_if=False)
            st6 = time.process_time()
            print(f"Time for IF: {st6-st5:.2f} s")
            if args.save_wandb:
                wandb.log({"Accuracy Time": st2-st1, "Coverage Time": st3-st2, "IWS Time": st4-st3,
                           "WeShap Time": st5-st4, "IF Time": st6-st5})
                wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Usage of WeShap
    parser.add_argument("--mode", type=str, default="denoise", choices=["denoise", "denoise-wl", "time"], help="Use Mode")
    # dataset
    parser.add_argument("--dataset-path", type=str, default="./data/wrench_data", help="dataset path")
    parser.add_argument("--dataset-name", type=str, default="youtube", help="dataset name")
    parser.add_argument("--feature-extractor", type=str, default="bert", help="feature for training end model")
    # label functions
    parser.add_argument("--lf-tag", type=str, default=None, help="tag of LF to use. If None, use original LFs")
    # weshap settings
    parser.add_argument("--weshap-distance-metric", type=str, default="euclidean",
                        choices=["euclidean", "manhattan", "cosine"], help="distance metric for WEShap")
    parser.add_argument("--weshap-k", type=int, default=10, help="k for WEShap")
    parser.add_argument("--weshap-weights", type=str, default="uniform", choices=["uniform", "distance"],
                        help="weight for WEShap")
    # influence function settings
    parser.add_argument("--if-type", type=str, default="if", choices=["if", "sif", "relatif"], help="type of IF")
    parser.add_argument("--if-mode", type=str, default="RW", choices=["RW", "WM", "normal"], help="mode of IF")
    parser.add_argument("--load-if", action="store_true", help="set to true if load precomputed IF")
    parser.add_argument("--save-if", action="store_true", help="set to true if save IF")
    # data programming
    parser.add_argument("--label-model", type=str, default="Snorkel", choices=["Snorkel", "MV"],
                        help="label model used in DP paradigm")
    parser.add_argument("--use-soft-labels", action="store_true",
                        help="set to true if use soft labels when training end model")
    parser.add_argument("--end-model", type=str, default="logistic", choices=["logistic", "mlp", "knn"],
                        help="end model in DP paradigm")
    parser.add_argument("--distance-metric", type=str, default="euclidean",
                        choices=["cosine", "euclidean", "manhatten"], help="distance metric for KNN")
    parser.add_argument("--abstain-strategy", type=str, default="discard", choices=["discard", "keep", "random"],
                        help="strategy for abstained data")
    parser.add_argument("--tune-label-model", action="store_true", help="tune label model hyperparameters")
    parser.add_argument("--tune-end-model", action="store_true", help="tune end model hyperparameters")
    parser.add_argument("--tune-metric", type=str, default="acc", choices=["acc", "f1"],
                        help="metric used to tune model hyperparameters")
    # experiments
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--save-wandb", action="store_true", help="save results to wandb")
    args = parser.parse_args()
    main(args)

