import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pyarrow.parquet as pq
import os

class StockConfig(object):
    def __init__(self):
        self.start_date = "2021-12-01"
        self.end_date = "2022-12-01"
        self.stock_list_file = "/data/quant/gaochao/stock_list.csv"
        self.daily_data_root = "/data/quant/gaochao/daily_data"
        self.min_data_root = "/data/quant/gaochao/min_data"
        self.trade_start_time = "09:30:00"
        self.trade_end_time = "15:00:00"

class StockData(Dataset):

    def __init__(self, config):
        self.config = config
        self._load_stock_list()
        self._load_data(self.stock_list)

    def _load_stock_list(self):
        self.stock_list_df = pd.read_csv(self.config.stock_list_file,
                                         dtype={"指数名称": str, "成分券代码": str, "成分券名称": str, "交易所": str},
                                         index_col="成分券代码")
        self.stock_list = self.stock_list_df.index.tolist()[:5]

    def _load_data(self, stock_list):
        self.data = list()
        for stock_code in stock_list:
            code_data = self._load_data_by_code(stock_code)
            self.data.extend(code_data)

    def _load_data_by_code(self, stock_code):
        stock_name = self.stock_list_df.loc[stock_code, "成分券名称"]
        file_name = "{}_{}.parquet".format(stock_code,stock_name)
        stock_daily_file = os.path.join(self.config.daily_data_root, file_name)
        stock_min_file = os.path.join(self.config.min_data_root, file_name)
        daily_df = self._process_daily_file(stock_daily_file)
        min_df = self._process_min_file(stock_min_file)
        code_data = self._construct_data(stock_code, daily_df, min_df)
        return code_data

    def _process_daily_file(self, file_path):
        df = pd.read_parquet(file_path)
        df["日期"] = pd.to_datetime(df["日期"])
        df["return_1"] = df["收盘"].pct_change(1)
        df["return_1"] = df["return_1"].shift(-1)
        df["return_2"] = df["收盘"].pct_change(2)
        df["return_2"] = df["return_1"].shift(-2)
        df.dropna()
        df = df.loc[df["日期"].between(self.config.start_date, self.config.end_date)]
        df.set_index("日期",inplace=True)
        df.sort_index(inplace=True)
        return df

    def _process_min_file(self, file_path):
        min_df = pd.read_parquet(file_path)
        min_df["tdate"] = pd.to_datetime(min_df["tdate"])
        min_df.set_index("tdate", inplace=True)
        min_df.sort_index(inplace=True)
        min_df = min_df.loc[self.config.start_date: self.config.end_date]
        return min_df

    def _construct_data(self, code, daily_df, min_df):
        data = list()
        unique_dates = sorted(list(set(min_df.index.date)))
        for date in unique_dates:
            today_data = min_df[min_df.index.date == pd.to_datetime(date).date()]
            today_data = today_data.between_time(self.config.trade_start_time, self.config.trade_end_time)
            today_label = daily_df[daily_df.index.date == pd.to_datetime(date).date()]
            record = {
                "input_data": today_data.to_numpy(),
                "date": date,
                "code": code,
                "return_1": float(today_label["return_1"].values),
                "return_2": float(today_label["return_2"].values)
            }
            data.append(record)
        return data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    cfg = StockConfig()
    data = StockData(cfg)
    for item in data:
        print(item['date'])
        print(item['code'])
    print("done")