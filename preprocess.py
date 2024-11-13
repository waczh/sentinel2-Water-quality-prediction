import os
import glob
import pandas as pd
from datetime import datetime
import chardet

def get_xlsx_files():
    return glob.glob("*.xlsx")

def extract_date_from_filename(filename):
    date_str = filename.split('_')[1][:8]
    return date_str

def read_data_file(data_file):

    with open(data_file, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    df = pd.read_csv(data_file, encoding=encoding)
    df.columns = df.columns.str.strip()
    df['监测时间'] = pd.to_datetime(df['监测时间'].str.replace('+', ' ', regex=False), format='%Y/%m/%d %H:%M:%S', errors='coerce')

    return df

def match_and_get_values(dataframe, target_date):

    matched_rows = dataframe[dataframe['监测时间'].dt.strftime('%Y%m%d') == target_date]
    if not matched_rows.empty:
        return matched_rows.iloc[1][[
            '水温', 'pH', '溶解氧', '电导率', '浊度',
            '高锰酸盐指数', '氨氮', '总磷', '总氮', '水质'
        ]].values

    return None

def write_to_xlsx(xlsx_files, data_folder):

    for xlsx_file in xlsx_files:
        date_str = extract_date_from_filename(xlsx_file)
        df_xlsx = pd.read_excel(xlsx_file)
        current_columns = df_xlsx.shape[1]
        if current_columns < 19:
            for _ in range(19 - current_columns):
                df_xlsx[f'新列{current_columns + 1}'] = None
        new_columns = ['水温', 'pH', '溶解氧', '电导率', '浊度',
                       '高锰酸盐指数', '氨氮', '总磷', '总氮', '水质']
        for col in new_columns:
            if col not in df_xlsx.columns:
                df_xlsx[col] = None
        csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
        for i, csv_file in enumerate(csv_files):
            df_data = read_data_file(csv_file)
            values = match_and_get_values(df_data, date_str)
            if values is not None:
                # 将匹配的值写入对应的行
                for j, value in enumerate(values):
                    df_xlsx.loc[i, new_columns[j]] = value
        df_xlsx.to_excel(xlsx_file, index=False)
        print(f"{xlsx_file} was transformed successfully!")

def match(data_folder):
    xlsx_files = get_xlsx_files()
    write_to_xlsx(xlsx_files, data_folder)

