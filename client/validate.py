import json
import os
import sys

from fedn.utils.helpers.helpers import save_metrics

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def get_metrics(in_model_path, out_json_path, data_path=None):
    """loads metrics json fiel and saves it via fedn, s.t. metrics can be sent to server."""
    with open('./metrics.json', 'r') as f:
        metrics = json.load(f)

    save_metrics(metrics, out_json_path)

if __name__ == "__main__":
    get_metrics(sys.argv[1], sys.argv[2])