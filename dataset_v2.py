import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pyarrow.parquet as pq
import os
import json
from tqdm import tqdm
import pickle
import numpy as np

class StockConfig(object):
    def __init__(self):
        self.start_date = "2021-01-01"
        self.end_date = "2021-12-31"
        self.stock_list_file = "/data/quant/stock_list.csv"
        self.daily_data_root = "/data/quant/exp/daily"
        self.min_data_root = "/data/quant/exp/2021-2022_tdx"
        self.trade_start_time = "09:30:00"
        self.trade_end_time = "15:00:00"
        self.normalize_range = 5
        self.load_from_cache = False
        self.load_with_write_cache = True
        self.cache_file = "/data/quant/exp/data.pickle"

class StockData(Dataset):

    def __init__(self, config):
        self.config = config
        if config.load_from_cache:
            self.load_pickle(config.cache_file)
        elif config.load_with_write_cache:
            self._load_with_save_cache()
        else:
            self._load_stock_list()
            self._load_data(self.stock_list)

    def _load_with_save_json(self):
        print("_load_with_save_json")
        self._load_stock_list()
        pbar = tqdm(total=len(self.stock_list))
        with open(self.config.json_file, "w") as f:
            for stock_code in self.stock_list:
                code_data = self._load_data_by_code(stock_code)
                for data_item in code_data:
                    json_line = json.dumps(data_item)
                    f.write(json_line + "\n")
                pbar.update(1)

    def _load_with_save_cache(self):
        print("_load_with_save_cache")
        self._load_stock_list()
        pbar = tqdm(total=len(self.stock_list))
        with open(self.config.cache_file, "wb") as f:
            for stock_code in self.stock_list:
                code_data = self._load_data_by_code(stock_code)
                for item in code_data:
                    pickle.dump(item, f)
                pbar.update(1)

    def _load_from_json(self, file_path):
        self.data = list()
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                item = json.loads(line)
                self.data.append(item)

    def _load_stock_list(self):
        self.stock_list_df = pd.read_csv(self.config.stock_list_file)
        self.stock_list = self.stock_list_df['code'].tolist()

    def _load_data(self, stock_list):
        self.data = list()
        pbar = tqdm(total=len(stock_list))
        for stock_code in stock_list:
            code_data = self._load_data_by_code(stock_code)
            self.data.extend(code_data)
            pbar.update(1)

    def _load_data_by_code(self, stock_code):
        min_file_name = "{}.parquet".format(stock_code)
        daily_file_name = "{}.csv".format(stock_code)
        stock_daily_file = os.path.join(self.config.daily_data_root, daily_file_name)
        stock_min_file = os.path.join(self.config.min_data_root, min_file_name)
        if not os.path.exists(stock_min_file):
            return []
        if not os.path.exists(stock_daily_file):
            return []
        daily_df = self._process_daily_file(stock_daily_file)
        min_df = self._process_min_file(stock_min_file)
        code_data = self._construct_data(stock_code, daily_df, min_df)
        return code_data

    def _process_daily_file(self, file_path):
        df = pd.read_csv(file_path)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["return_1"] = df["close"].pct_change(1)
        df["return_1"] = df["return_1"].shift(-1)
        df["return_2"] = df["close"].pct_change(2)
        df["return_2"] = df["return_2"].shift(-2)
        df.dropna()
        df = df.loc[df["trade_date"].between(self.config.start_date, self.config.end_date)]
        df.set_index("trade_date",inplace=True)
        df.sort_index(inplace=True)
        return df

    def _process_min_file(self, file_path):
        min_df = pd.read_parquet(file_path)
        min_df["日期"] = pd.to_datetime(min_df["日期"])
        min_df.set_index("日期",inplace=True)
        min_df.sort_index(inplace=True)
        min_df = min_df.loc[self.config.start_date: self.config.end_date]
        self._process_normalize(min_df)
        min_df.dropna(inplace = True)
        return min_df

    def _process_normalize(self, df):

        def get_min_max_price(df):
            min_open = df["开盘价"].min()
            min_close = df["收盘价"].min()
            min_low = df["最低价"].min()
            min_high = df["最高价"].min()
            min_price= min([min_open, min_low, min_high, min_close])

            max_open = df["开盘价"].max()
            max_close = df["收盘价"].max()
            max_low = df["最低价"].max()
            max_high= df["最高价"].max()
            max_price = max([max_open, max_low, max_high, max_close])

            return min_price, max_price

        def get_min_max_volume(df):
            return df["成交量"].min(), df["成交量"].max()

        def get_min_max_amount(df):
            return df["成交额"].min(), df["成交额"].max()

        def normalize(df, dateindex, column_normalize ,column_name, min_value, max_value):
            dateindex = dateindex.strftime("%Y-%m-%d")
            df.loc[dateindex, column_normalize] = (df.loc[dateindex][column_name] - min_value)/(max_value - min_value)

        window = self.config.normalize_range
        unique_dates = sorted(list(set(df.index.date)))
        for index,date in enumerate(unique_dates):
            if index < window:
                continue
            start_index = index - window
            end_index = index
            slice_df = df.loc[unique_dates[start_index]:unique_dates[end_index]]
            min_price, max_price = get_min_max_price(slice_df)
            min_volume, max_volume = get_min_max_volume(slice_df)
            min_amount, max_amount = get_min_max_amount(slice_df)
            normalize(df, date, "开盘价_N", "开盘价", min_price, max_price)
            normalize(df, date, "最低价_N", "最低价", min_price, max_price)
            normalize(df, date, "最高价_N", "最高价", min_price, max_price)
            normalize(df, date, "收盘价_N", "收盘价", min_price, max_price)
            normalize(df, date, "成交量_N", "成交量", min_volume, max_volume)
            normalize(df, date, "成交额_N", "成交额", min_amount, max_amount)

    def _construct_data(self, code, daily_df, min_df):
        data = list()
        columns = ["开盘价_N","最高价_N","最低价_N", "收盘价_N","成交量_N","成交额_N"]
        unique_dates = sorted(list(set(min_df.index.date)))
        for date in unique_dates:
            try:
                today_data = min_df[min_df.index.date == pd.to_datetime(date).date()]
                today_data = today_data.reset_index(drop=True)
                today_data["时间"] = pd.to_datetime(today_data["时间"])
                today_data.set_index("时间", inplace=True)
                today_data.sort_index(inplace=True)
                today_data = today_data.between_time(self.config.trade_start_time, self.config.trade_end_time)
                today_data = today_data[columns]
                today_label = daily_df[daily_df.index.date == pd.to_datetime(date).date()]
                if len(today_label) == 0:
                    continue
                record = {
                    #"input_data": today_data.values.tolist(),
                    "input_data": today_data.to_numpy(dtype=np.float32),
                    "date": str(date),
                    "code": code,
                    "return_1": float(today_label["return_1"].values),
                    "return_2": float(today_label["return_2"].values)
                }
                data.append(record)
            except Exception as e:
                print(e)
        return data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def to_json(self, file_path):
        print("to_json total data count: {}".format(len(self.data)))
        with open(file_path, "w") as f:
           for data_item in self.data:
               json_line = json.dumps(data_item)
               f.write(json_line+"\n")

    def to_pickle(self, file_path):
        with open(file_path, "wb") as fp:
            pickle.dump(self.data, fp)

    def load_pickle(self, file_path):
        self.data = list()
        with open(file_path, "rb") as fp:
            while True:
                try:
                    self.data.append(pickle.load(fp))
                except EOFError:
                    break

    def min_file_exist_check(self):
        for stock_code in self.stock_list:
            stock_name = self.stock_list_df[self.stock_list_df['成分券代码'] == stock_code]["成分券名称"].values[0]
            stock_name = stock_name.replace(" ", "")
            file_name = "{}_{}.parquet".format(stock_code, stock_name)
            if not os.path.exists(os.path.join(self.config.min_data_root,file_name)):
                print(file_name)

    def daily_file_exist_check(self):
        for stock_code in self.stock_list:
            stock_name = self.stock_list_df[self.stock_list_df['成分券代码'] == stock_code]["成分券名称"].values[0]
            stock_name = stock_name.replace(" ", "")
            file_name = "{}_{}.parquet".format(stock_code, stock_name)
            if not os.path.exists(os.path.join(self.config.daily_data_root,file_name)):
                print(file_name)

if __name__ == "__main__":
    cfg = StockConfig()
    data = StockData(cfg)
    print("load finish")
    # for item in data:
    #     print(item['date'])
    #     print(item['code'])
    #     print(item['return_1'])
    # print("done")
    # print(len(data))