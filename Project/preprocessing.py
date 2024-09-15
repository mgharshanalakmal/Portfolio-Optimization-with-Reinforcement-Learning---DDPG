import pandas as pd
import os
import time


def convert_value(value):
    unit = str(value)[-1]
    numeric_part = float(str(value)[:-1])
    if unit == "K":
        return numeric_part * 1000
    elif unit == "M":
        return numeric_part * 1000000
    else:
        return numeric_part


if __name__ == "__main__":
    folder_path = "../Data"

    start_time = time.time()

    main_df = pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path).dropna()

            df["Volume"] = df["Vol."].apply(convert_value)
            df = df.drop("Vol.", axis=1)
            df["Change %"] = (df["Change %"].str[:-1].astype(float)) / 100
            df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
            df = df.sort_values(by=["Date"], ascending=True)

            df["vh"] = df["High"] / df["Open"]
            df["vl"] = df["Low"] / df["Open"]
            df["dvol"] = (df["High"] - df["Low"]) / df["Price"]

            value_columns = [
                "Price",
                "Open",
                "High",
                "Low",
                "Change %",
                "Volume",
                "vh",
                "vl",
                "dvol",
            ]
            for column in value_columns:
                df[f"{column}_s"] = df[column].shift(1)
                df[f"{column}_roll_7"] = df[column].rolling(7).mean()
                df[f"{column}_roll_14"] = df[column].rolling(14).mean()
                df[f"{column}_roll_21"] = df[column].rolling(21).mean()
                df[f"{column}_roll_28"] = df[column].rolling(28).mean()

            file = os.path.splitext(filename)[0]
            column_list = [f"{file}_{column}" for column in df.columns]
            df.columns = column_list

            if file == "ACL":
                date_col_name = df.columns[0]
                df = df.rename(columns={date_col_name: "Date"})
            else:
                date_col_name = df.columns[0]
                df = df.drop(date_col_name, axis=1)

            main_df = pd.concat([main_df, df], axis=1).dropna()

    main_df.to_json("Data/data.json", orient="records")
    main_df.to_csv("Data/data.csv", index=False)

    end_time = time.time()

    print(f"Execution time {end_time - start_time}")
