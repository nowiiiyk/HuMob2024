import multiprocessing as mp

import pandas as pd

from linear_interpolater import LinearInterpolater

# 並列処理のために、線形補完メソッドのラッパー関数を定義
li = LinearInterpolater()
def linear_interpolate_wrapper(args):
    df, interpolate_n, d_range, t_range = args
    return li.linear_interpolate(df, interpolate_n, d_range, t_range)

# ---------------------------- 実処理 -----------------------------------
def main():
    # ---------------------------- 定数定義 -----------------------------------
    # 教師データの区間数
    D_TRAIN_RANGE = 60
    
    # 1日を30分ごとの時間帯に分割したときの区間数
    T_RANGE = 48
    
    # 補完対象とする欠損ブロックの最大長(ex.N=8ならば8時間連続して欠損した場合に線形補完)
    INTERPOLATE_N = 24

    # 入出力パス
    INPUT_PATH = "../../data/city_B_challengedata.csv.gz"
    OUTPUT_PATH = f"../../data/city_B_challengedata_interpolated_n{INTERPOLATE_N}.csv"
    
    # ---------------------------- データ読み込み・分割 -----------------------------------
    df = pd.read_csv(INPUT_PATH)
    
    # HuMob2025のデータはdが1から始まるため補正
    df["d"] = df["d"] - 1
    
    df_train = (
        df
        .loc[df["d"] < D_TRAIN_RANGE]
    )
    
    df_pred = (
        df
        .loc[df["d"] >= D_TRAIN_RANGE]
    )

    # ---------------------------- 線形補完（並列処理） -----------------------------------
    grouped = [group for _, group in df_train.groupby("uid")]
    list_args = [(g, INTERPOLATE_N, D_TRAIN_RANGE, T_RANGE) for g in grouped]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        list_results = pool.map(linear_interpolate_wrapper, list_args)

    df_result = pd.concat(list_results)

    # ---------------------------- データ保存 -----------------------------------
    df_full = pd.concat([df_result, df_pred], axis=0,  ignore_index=True)

    # HuMob2025のデータはdが1から始まるため補正
    df_full["d"] = df_full["d"] + 1

    df_full.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()