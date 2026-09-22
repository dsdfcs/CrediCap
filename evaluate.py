import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


ROOT = Path("/home/xgd/FLEUR_reproduction/01_source_code/FLEUR")
ANN = ROOT / "annotations"


def read_predictions(path):
    """
    从 RefFLEUR result txt 中读取:
        our score : ...
    """
    values = []

    score_pat = re.compile(
        r"^our score\s*:\s*(.*)$",
        re.IGNORECASE
    )

    num_pat = re.compile(
        r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?"
    )

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        for line in f:

            m = score_pat.match(line.strip())

            if not m:
                continue

            n = num_pat.search(m.group(1))

            if n:
                values.append(float(n.group(0)))

    return np.asarray(values, dtype=np.float64)


def gold_expert():
    """
    Flickr8k-Expert:
    每个 candidate 有3个 expert rating，范围1~4。

    先求3个人平均:
        mean(r1,r2,r3)

    再归一化:
        (score - 1) / 3
    """

    with open(
        ANN / "flickr8k.json",
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    gold = []

    for _, v in data.items():

        valid = []

        for h in v["human_judgement"]:

            r = float(h["rating"])

            if math.isnan(r):
                continue

            valid.append(h)

        if len(valid) % 3 != 0:
            raise RuntimeError(
                "Expert annotation 不是3个评分为一组"
            )

        for i in range(0, len(valid), 3):

            block = valid[i:i + 3]

            captions = {
                " ".join(x["caption"].split())
                for x in block
            }

            if len(captions) != 1:
                raise RuntimeError(
                    "Expert同一组三个评分对应的caption不同"
                )

            human_mean = np.mean(
                [float(x["rating"]) for x in block]
            )

            human_norm = (
                human_mean - 1.0
            ) / 3.0

            gold.append(human_norm)

    return np.asarray(
        gold,
        dtype=np.float64
    )


def gold_cf():
    """
    Flickr8k-CF:
    annotation 中 rating 本身已经是
    0~1 的人工 yes 比例。
    """

    with open(
        ANN / "crowdflower_flickr8k.json",
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    gold = []

    for _, v in data.items():

        for h in v["human_judgement"]:

            r = float(h["rating"])

            if math.isnan(r):
                continue

            gold.append(r)

    return np.asarray(
        gold,
        dtype=np.float64
    )


def gold_composite():
    """
    COMPOSITE:
        human score = 1~5

    归一化:
        (score - 1) / 4
    """

    with open(
        ANN / "composite.json",
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    gold = []

    for _, rows in data.items():

        for sample in rows:

            raw = float(
                sample["human"]
            )

            norm = (
                raw - 1.0
            ) / 4.0

            gold.append(norm)

    return np.asarray(
        gold,
        dtype=np.float64
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "expert",
            "cf",
            "composite"
        ]
    )

    parser.add_argument(
        "--result",
        required=True
    )

    args = parser.parse_args()

    pred = read_predictions(
        args.result
    )

    if args.dataset == "expert":

        gold = gold_expert()
        expected = 5664
        native_factor = 3.0

    elif args.dataset == "cf":

        gold = gold_cf()
        expected = 47830
        native_factor = 1.0

    else:

        gold = gold_composite()
        expected = 11985
        native_factor = 4.0

    if len(gold) != expected:
        raise RuntimeError(
            f"Gold数量异常: "
            f"{len(gold)} != {expected}"
        )

    print(
        f"dataset={args.dataset}"
    )

    print(
        f"pred_N={len(pred)}"
    )

    print(
        f"gold_N={len(gold)}"
    )

    if len(pred) != len(gold):

        raise RuntimeError(
            "RefFLEUR预测数量和人工真值数量不一致。\n"
            "说明本次推理没有完整跑完，"
            "或者某些样本发生了评分解析错误。\n"
            "禁止在这种情况下计算MAE/RMSE。"
        )

    if np.any(pred < 0) or np.any(pred > 1):

        raise RuntimeError(
            "存在不在[0,1]范围内的RefFLEUR预测"
        )

    diff = pred - gold

    mae = float(
        np.mean(
            np.abs(diff)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                np.square(diff)
            )
        )
    )

    bias = float(
        np.mean(diff)
    )

    print(
        f"MAE={mae:.6f}"
    )

    print(
        f"RMSE={rmse:.6f}"
    )

    print(
        f"BIAS={bias:.6f}"
    )

    print(
        f"PRED_MEAN={np.mean(pred):.6f}"
    )

    print(
        f"GOLD_MEAN={np.mean(gold):.6f}"
    )

    # 只是方便理解，论文主表建议用上面的 normalized 指标
    if native_factor != 1.0:

        print(
            f"NATIVE_MAE={mae * native_factor:.6f}"
        )

        print(
            f"NATIVE_RMSE={rmse * native_factor:.6f}"
        )


if __name__ == "__main__":
    main()
