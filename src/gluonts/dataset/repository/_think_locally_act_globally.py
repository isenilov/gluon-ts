# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import json
import os
from pathlib import Path

import pandas as pd

from gluonts.dataset.repository._lstnet import LstnetDataset, load_from_pandas
from gluonts.dataset.repository._util import metadata, save_to_file, to_dict
import numpy as np

"""
to download the datasets:
#!/bin/bash
function gdrive-get() {
    fileid=$1
    filename=$2
    if [[ "${fileid}" == "" || "${filename}" == "" ]]; then
        echo "gdrive-curl gdrive-url|gdrive-fileid filename"
        return 1
    else
        if [[ ${fileid} = http* ]]; then
            fileid=$(echo ${fileid} | sed "s/http.*drive.google.com.*id=\([^&]*\).*/\1/")
        fi
        echo "Download ${filename} from google drive with id ${fileid}..."
        cookie="/tmp/cookies.txt"
        curl -c ${cookie} -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
        confirmid=$(awk '/download/ {print $NF}' ${cookie})
        curl -Lb ${cookie} "https://drive.google.com/uc?export=download&confirm=${confirmid}&id=${fileid}" -o ${filename}
        rm -rf ${cookie}
        return 0
    fi
}

gdrive-get 1fkK538QhyQUvnb6w_rCp6iQU1Yc-wQ8E electricity.npy
gdrive-get 1JAFE7PzZTVw4rwttkxOPGtIQyfICjNBy traffic.npy
gdrive-get 1xEDHyAdY2VbFJ-oD5sH1nd33wClaoNyC wiki.npy

#python reshape_data.py 
"""

npy_folder = Path("/Users/dsalina/Documents/Code/evaluate-think-globally")

datasets_info = {
    "electricity-glo": LstnetDataset(
        name="electricity-glo",
        url=npy_folder / "electricity.npy",
        # original dataset can be found at https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014#
        # the aggregated ones that is used from LSTNet filters out from the initial 370 series the one with no data
        # in 2011
        num_series=370,
        num_time_steps=26136,
        prediction_length=24,
        rolling_evaluations=7,
        start_date='2012-01-01',
        freq='1H',
        agg_freq='1H',
    ),
    "traffic-glo": LstnetDataset(
        name="traffic-glo",
        url=npy_folder / "traffic.npy",
        num_series=963,
        num_time_steps=10560,
        prediction_length=24,
        rolling_evaluations=7,
        start_date='2015-01-01',
        freq='1H',
        agg_freq='1H',
    ),
}


def generate_glo_dataset(dataset_path: Path, dataset_name: str):
    ds_info = datasets_info[dataset_name]

    os.makedirs(dataset_path, exist_ok=True)

    with open(dataset_path / 'metadata.json', 'w') as f:
        f.write(
            json.dumps(
                metadata(
                    cardinality=ds_info.num_series,
                    freq=ds_info.freq,
                    prediction_length=ds_info.prediction_length,
                )
            )
        )

    train_file = dataset_path / "train" / "data.json"
    test_file = dataset_path / "test" / "data.json"

    time_index = pd.date_range(
        start=ds_info.start_date,
        freq=ds_info.freq,
        periods=ds_info.num_time_steps,
    )

    # (N, T)
    data = np.load(ds_info.url)

    df = pd.DataFrame(data).T

    assert df.shape == (
        ds_info.num_time_steps,
        ds_info.num_series,
    ), f"expected num_time_steps/num_series {(ds_info.num_time_steps, ds_info.num_series)} but got {df.shape}"

    timeseries = load_from_pandas(
        df=df, time_index=time_index, agg_freq=ds_info.agg_freq
    )

    # the last date seen during training
    ts_index = timeseries[0].index

    training_end = ts_index[-168]
    # training_end = ts_index[int(len(ts_index) * (8 / 10))]

    train_ts = []
    for cat, ts in enumerate(timeseries):
        sliced_ts = ts[:training_end]
        if len(sliced_ts) > 0:
            train_ts.append(
                to_dict(
                    target_values=sliced_ts.values,
                    start=sliced_ts.index[0],
                    cat=[cat],
                )
            )

    assert len(train_ts) == ds_info.num_series

    save_to_file(train_file, train_ts)

    # time of the first prediction
    prediction_dates = [
        training_end + i * ds_info.prediction_length
        for i in range(ds_info.rolling_evaluations)
    ]

    test_ts = []
    for prediction_start_date in prediction_dates:
        for cat, ts in enumerate(timeseries):
            # print(prediction_start_date)
            prediction_end_date = (
                prediction_start_date + ds_info.prediction_length
            )
            sliced_ts = ts[:prediction_end_date]
            test_ts.append(
                to_dict(
                    target_values=sliced_ts.values,
                    start=sliced_ts.index[0],
                    cat=[cat],
                )
            )

    assert len(test_ts) == ds_info.num_series * ds_info.rolling_evaluations

    save_to_file(test_file, test_ts)


if __name__ == '__main__':
    generate_glo_dataset(Path("/tmp/"), "electricity-glo")