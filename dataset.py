import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pyarrow.parquet as pq
import os
import json
from tqdm import tqdm

class StockConfig(object):
    def __init__(self):
        self.start_date = "2021-12-01"
        self.end_date = "2022-12-01"
        self.stock_list_file = "/data/quant/gaochao/stock_list.csv"
        self.daily_data_root = "/data/quant/gaochao/daily_data_v1"
        self.min_data_root = "/data/quant/gaochao/min_data"
        self.trade_start_time = "09:30:00"
        self.trade_end_time = "15:00:00"
        self.json_file = "/data/quant/gaochao/data.json"
        self.load_from_json = True
        self.load_with_write_json = False

class StockData(Dataset):

    def __init__(self, config):
        self.config = config
        if config.load_from_json:
            self._load_from_json(config.json_file)
        elif config.load_with_write_json:
            self._load_with_save_json()
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

    def _load_from_json(self, file_path):
        self.data = list()
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                item = json.loads(line)
                self.data.append(item)

    def _load_stock_list(self):
        self.stock_list_df = pd.read_csv(self.config.stock_list_file,
                                         dtype={"指数名称": str, "成分券代码": str, "成分券名称": str, "交易所": str})
        self.stock_list_df['成分券代码'].apply(lambda x: ':0>6d'.format(x))
        self.stock_list = self.stock_list_df['成分券代码'].tolist()

    def _load_data(self, stock_list):
        self.data = list()
        pbar = tqdm(total=len(stock_list))
        for stock_code in stock_list:
            code_data = self._load_data_by_code(stock_code)
            self.data.extend(code_data)
            pbar.update(1)

    def _load_data_by_code(self, stock_code):
        stock_name = self.stock_list_df[self.stock_list_df['成分券代码'] == stock_code]["成分券名称"].values[0]
        stock_name = stock_name.replace(" ","")
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

    def _normalize(self, df, columns):
        def min_max_scaling(series):
            return (series - series.min()) / (series.max() - series.min())

        for column in columns:
            if column in df.columns:
                df[column] = min_max_scaling(df[column])
        return df

    def _construct_data(self, code, daily_df, min_df):
        data = list()
        columns = ["open", "close", "high", "low", "cjl","cje","cjjj"]
        unique_dates = sorted(list(set(min_df.index.date)))
        for date in unique_dates:
            try:
                today_data = min_df[min_df.index.date == pd.to_datetime(date).date()]
                today_data = today_data.between_time(self.config.trade_start_time, self.config.trade_end_time)
                today_data = self._normalize(today_data, columns)
                today_data = today_data[columns]
                today_label = daily_df[daily_df.index.date == pd.to_datetime(date).date()]
                if len(today_label) == 0:
                    continue
                record = {
                    "input_data": today_data.values.tolist(),
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
    for item in data:
        print(item['date'])
        print(item['code'])
        print(item['return_1'])
    #data.to_json("/data/quant/gaochao/data.json")
    #data.min_file_exist_check()
    #data.daily_file_exist_check()
    # for item in data:
    #     print(item['date'])
    #     print(item['code'])
    #     print(item['return_1'])
    # print("done")