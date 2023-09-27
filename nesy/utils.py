import torch
import numpy as np
import random
import json
import os
from pathlib import Path


def set_seed(seed):
    """
    It sets the seed for the reproducibility of the experiments.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_u_g_matrix(mf_model, threshold):
    """
    It computes the user X genre matrix G using the given model to get the predictions. It also binarizes it using
    the given threshold.

    :param mf_model: Matrix Factorization model that has to be used to construct the matrix
    :param threshold: threshold used as decision boundary
    :return: matrix G
    """
    matrix = torch.matmul(mf_model.u_emb.weight.data, torch.t(mf_model.i_emb.weight.data))
    matrix = torch.add(matrix, mf_model.u_bias.weight.data)
    matrix = torch.add(matrix, torch.t(mf_model.i_bias.weight.data))
    matrix = torch.sigmoid(matrix)
    matrix = torch.where(matrix >= threshold, 1., 0.)
    return matrix


def generate_report_dict(results_path, metrics):
    """
    It generates a report dictionary containing the results of the experiment.
    Every metric is averaged across the runs. Mean and standard deviation are reported for each metric.
    It saves the dictionary as a JSON file in the root folder where the result files are picked.

    :param results_path: path where to take the results of the experiments
    :param metrics: metrics that have to be included in the table
    :return:
    """
    # create result dict
    result_dict = {}
    # iterate over result files in the directory
    files = Path(results_path).glob('*')
    for file in files:
        if "-genres-" not in file.name:
            # get file metadata
            metadata = file.name.split("-")
            model_name = metadata[0]
            p = metadata[5].replace(".json", "")
            with open(file) as result_file:
                res_dict = json.load(result_file)
            if model_name not in result_dict:
                result_dict[model_name] = {p: {k: [v] for k, v in res_dict.items() if k in metrics}}
            elif p not in result_dict[model_name]:
                result_dict[model_name][p] = {k: [v] for k, v in res_dict.items() if k in metrics}
            else:
                for k in result_dict[model_name][p]:
                    result_dict[model_name][p][k].append(res_dict[k])

    for model in result_dict:
        for p in result_dict[model]:
            for k in result_dict[model][p]:
                result_dict[model][p][k] = str(np.mean(result_dict[model][p][k])) + " +/- " + str(np.std(result_dict[model][p][k]))

    with open(os.path.join(results_path.split("/")[0], "results.json"), 'w') as fp:
        json.dump(result_dict, fp, indent=4)


def generate_report_table(report_path, metric_dict, model_dict, models=None):
    """
    Generate a report LaTeX table given a report json file.

    :param report_path: path to the report json file.
    :param metric_dict: dictionary to give the desired names to the metrics in the table
    :param model_dict: dictionary to give the desired names to the models in the table
    :param models: models for which the table has to be generated
    """
    with open(report_path) as f:
        report = json.load(f)

    if models is None:
        models = list(report.keys())
    folds = sorted(list(report[models[0]].keys()), reverse=True)
    metrics = list(report[models[0]][folds[0]].keys())

    report_dict = {fold: {metric: [] for metric in metrics} for fold in folds}

    for model in models:
        for fold in folds:
            for metric in metrics:
                report_dict[fold][metric].append((round(float(report[model][fold][metric].split(" +/- ")[0]), 4),
                                                  round(float(report[model][fold][metric].split(" +/- ")[1]), 4)))

    max_fold_metric = {fold: {metric: max([metric_mean for metric_mean, _ in report_dict[fold][metric]])
                              for metric in metrics}
                       for fold in folds}

    table = "\\begin{table*}[ht!]\n\\centering\n\\small\n\\begin{tabular}{ l | l | " + " | ".join(["c" for _ in models]) + " }\n"
    table += "Fold & Metric & " + " & ".join([model_dict[model] for model in models]) + "\\\\\n\\hline"
    for fold in report_dict:
        table += "\n\\multirow{%d}{*}{%d\%%}" % (len(metrics), float(fold) * 100)
        for metric in metrics:
            table += " &" + (" %s & " + " & ".join([("%.4f$_{(%.4f)}$" if mean_metric != max_fold_metric[fold][
                metric] else "\\textbf{%.4f}$_{(%.4f)}$") % (mean_metric, variance_metric) for
                                                    mean_metric, variance_metric in
                                                    report_dict[fold][metric]])) % metric_dict[metric] + "\\\\\n"
        table += "\\hline"

    table += "\n\\end{tabular}\n\\caption{Test metrics}\n\\end{table*}"
    return table
