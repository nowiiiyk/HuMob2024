"""
複数ユーザで構成されたdataframeをインプットに線形補完を行うよう設計されたクラス
一方で、11_run_linear_interpolation.pyでは高速化のため、単一ユーザごとに本クラスを呼び出し並列実行している
そのため、本クラスに含まれるユーザごとの処理は本来不要だが、処理速度に大きな影響がないことと、対応工数都合でリファクタを見送り
ex.) for uid, df_user in df_train_full_st.groupby("uid"):など
"""

import pandas as pd

class LinearInterpolater:
    """
    線形補完を行うためのメソッドを集約したクラス
    当初、複数ユーザで構成されたdataframeをインプットに線形補完を行うよう設計していた
    一方で、11_run_linear_interpolation.pyでは高速化のため、単一ユーザごとに本クラスを呼び出し並列実行している
    そのため、本クラスに含まれるユーザごとの処理は本来不要だが、処理速度に大きな影響がないことと、対応工数都合でリファクタを見送り
    ex.) for uid, df_user in df_train_full_st.groupby("uid"):など
    """

    def __init__(self):
        pass

    def linear_interpolate(self, df_train, interpolate_n=8, d_range=60, t_range=48):
        """教師データについて、欠損しているレコードを線形補完"""

        # 教師データについて、欠損しているレコードをNullで復元
        df_train_full = self._get_full_df(df_train, d_range, t_range)

        # ユーザごとに日付・時刻順にソートし、各レコードに0から始まる連番を付与（日付を跨いでも加算され続けるt）
        df_train_full_st = self._add_sequential_time(df_train_full)

        # 線形補完
        df_train_fill = self._lerp_df(df_train_full_st, interpolate_n)

        return df_train_fill
        
    def _get_full_df(self, df_train, d_range, t_range):
        """教師データについて、欠損しているレコードを(x, y) = (-1, -1)で復元"""

        # ユニークなuid, d, tをそれぞれdfとして作成
        df_uid = pd.DataFrame({"uid": df_train["uid"].unique()})
        df_d = pd.DataFrame({"d": range(0, d_range)})
        df_t = pd.DataFrame({"t": range(0, t_range)})

        # ①：作成したdfをクロス結合し、全てのuid, d, tの組み合わせを保持するdfを作成
        # ②：教師データに存在するレコードは(x, y)を該当する値で、欠損しているレコードは(x, y) = (-1, -1)で復元
        df_train_full = (
            # ①
            df_uid.assign(key=1)
            .merge(df_d.assign(key=1), on='key')
            .merge(df_t.assign(key=1), on='key')
            .drop('key', axis=1)

            # ②
            .merge(df_train, on=['uid', 'd', 't'], how='left')
            .fillna(-1).copy()
            .sort_values(by=['uid', 'd', 't'])
        )

        return df_train_full

    def _add_sequential_time(self, df_train_full):
        """ユーザごとに日付・時刻順にソートし、各レコードに0から始まる連番を付与"""
            
        df_with_seq_t = df_train_full.sort_values(["uid", "d", "t"]).copy()
        df_with_seq_t["sequential_t"] = (
            df_with_seq_t
            .groupby("uid")
            .cumcount()
        )

        return df_with_seq_t

    def _lerp_df(self, df_train_full_st, interpolate_n):
        """ユーザごとに欠損レコードに対して線形補完を行い、補完後のデータを縦結合"""

        # ユーザごとに線形補完したデータを格納するリスト
        df_filled_list = []

        # 全てのユーザをループ
        for uid, df_user in df_train_full_st.groupby("uid"):
            df_user = df_user.sort_values("sequential_t").copy().reset_index(drop=True)
    
            df_user_filled = self._fill_missing_for_user(df_user, interpolate_n)
            df_filled_list.append(df_user_filled)
    
        df_train_fill = (
            pd.concat(df_filled_list)
            .sort_values(["uid", "sequential_t"])
            .reset_index(drop=True)
            .drop(columns=["sequential_t"])
        )
        return df_train_fill

    def _fill_missing_for_user(self, df_user, interpolate_n):
        """ユーザーの時系列データに対して、欠損レコードを線形補完"""

        # 全てのレコードをループ
        for i in range(len(df_user)):

            # 該当レコードが欠損している場合
            if df_user.iloc[i]["x"] == -1:

                # 線形補完に用いるレコードを抽出
                prev_valid, next_valid = self._find_surrounding_valid_points(df_user, i)

                # 線形補完に用いるレコードとして、有効なデータが抽出できた場合
                if prev_valid is not None and next_valid is not None:

                    # レコード間の時間の差を算出、基準値以内であれば線形補完を実施
                    time_diff = int(next_valid["sequential_t"] - prev_valid["sequential_t"])
                    if time_diff < interpolate_n * 2:
                        self._apply_linear_interpolation(df_user, prev_valid, next_valid, i, time_diff)

        # 線形補完後も欠損しているレコードを除外
        df_user_fill = df_user[df_user["x"] != -1].copy()
        df_user_fill["x"] = df_user_fill["x"].astype(int)
        df_user_fill["y"] = df_user_fill["y"].astype(int)
        
        return df_user_fill

    def _find_surrounding_valid_points(self, df_user, i):
        """指定インデックスの前後に存在する線形補完に有効なレコードを探索"""
        prev_valid = None
        next_valid = None
    
        if (i != 0) and (df_user.iloc[i - 1]["x"] != -1):
            prev_valid = df_user.iloc[i - 1]
    
        for j in range(i + 1, len(df_user)):
            if df_user.iloc[j]["x"] != -1:
                next_valid = df_user.iloc[j]
                break
    
        return prev_valid, next_valid

    def _apply_linear_interpolation(self, df_user, prev_valid, next_valid, start_idx, time_diff):
        """欠損ブロックに対して、前後の有効データを用いて線形補完"""
        x_diff = next_valid["x"] - prev_valid["x"]
        y_diff = next_valid["y"] - prev_valid["y"]
    
        for j in range(start_idx, start_idx + time_diff - 1):
            if df_user.iloc[j]["x"] == -1:
                time_from_prev = df_user.iloc[j]["sequential_t"] - prev_valid["sequential_t"]
                df_user.iat[j, df_user.columns.get_loc("x")] = prev_valid["x"] + (x_diff / time_diff) * time_from_prev
                df_user.iat[j, df_user.columns.get_loc("y")] = prev_valid["y"] + (y_diff / time_diff) * time_from_prev

        return 0