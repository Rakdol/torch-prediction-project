import os
import time
from argparse import ArgumentParser
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import psycopg2


CREATE_LOAD_SITE = """
CREATE TABLE IF NOT EXISTS load_site (
    id SERIAL PRIMARY KEY,
    load_code TEXT UNIQUE NOT NULL,
    location TEXT,
    description TEXT
);
"""

CREATE_SOLAR_SITE = """
CREATE TABLE IF NOT EXISTS solar_site (
    id SERIAL PRIMARY KEY,
    solar_code TEXT UNIQUE NOT NULL,
    load_id INTEGER NOT NULL,
    panel_capacity_kw FLOAT,
    FOREIGN KEY (load_id) REFERENCES load_site(id)
);
"""

CREATE_LOAD_DATA = """
CREATE TABLE IF NOT EXISTS load_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    load_id INTEGER NOT NULL,
    demand FLOAT,
    FOREIGN KEY (load_id) REFERENCES load_site(id),
    UNIQUE (timestamp, load_id)
);
"""

CREATE_SOLAR_DATA = """
CREATE TABLE IF NOT EXISTS solar_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    solar_id INTEGER NOT NULL,
    generation FLOAT,
    FOREIGN KEY (solar_id) REFERENCES solar_site(id),
    UNIQUE (timestamp, solar_id)
);
"""

CREATE_LOAD_WEATHER = """
CREATE TABLE IF NOT EXISTS load_weather (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    load_id INTEGER NOT NULL,
    temperature FLOAT,
    humidity FLOAT,
    FOREIGN KEY (load_id) REFERENCES load_site(id),
    UNIQUE (timestamp, load_id)
);
"""

CREATE_SOLAR_WEATHER = """
CREATE TABLE IF NOT EXISTS solar_weather (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    solar_id INTEGER NOT NULL,
    temperature FLOAT,
    humidity FLOAT,
    wind_speed FLOAT,
    wind_direction FLOAT,
    cloud_cover VARCHAR(20),
    FOREIGN KEY (solar_id) REFERENCES solar_site(id),
    UNIQUE (timestamp, solar_id)
);
"""


def df_to_insert_sql(df, table_name, column_order=None):
    """DataFrame을 INSERT INTO SQL 문자열로 변환"""
    if column_order:
        df = df[column_order]

    values = []
    for row in df.itertuples(index=False, name=None):
        formatted_row = []
        for item in row:
            if pd.isna(item):
                formatted_row.append("NULL")
            elif isinstance(item, str):
                formatted_row.append(f"'{item}'")
            elif isinstance(item, pd.Timestamp):
                formatted_row.append(f"'{item.strftime('%Y-%m-%d %H:%M:%S')}'")
            else:
                formatted_row.append(str(item))
        values.append(f"({', '.join(formatted_row)})")

    columns = column_order if column_order else df.columns
    sql = (
        f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES\n"
        + ",\n".join(values)
        + ";"
    )
    return sql


def insert_load(load_data):
    """load_data DataFrame을 PostgreSQL에 삽입"""
    insert_sql = df_to_insert_sql(
        load_data, "load_data", [timestamp_col, "load_id", "demand"]
    )
    try:
        with db_connect.cursor() as cur:
            cur.execute(insert_sql)
            db_connect.commit()
    except Exception as e:
        db_connect.rollback()
        print("[ERROR] SQL 실행 중 오류 발생:", e)
        print("실행 SQL:\n", insert_sql[:500], "...")  # 너무 길면 일부만 출력


def insert_load_weather(load_data):
    """load_weather DataFrame을 PostgreSQL에 삽입"""
    insert_sql = df_to_insert_sql(
        load_data, "load_weather", [timestamp_col, "load_id", "temperature", "humidity"]
    )
    try:
        with db_connect.cursor() as cur:
            cur.execute(insert_sql)
            db_connect.commit()
    except Exception as e:
        db_connect.rollback()
        print("[ERROR] SQL 실행 중 오류 발생:", e)
        print("실행 SQL:\n", insert_sql[:500], "...")  # 너무 길면 일부만 출력


def insert_solar(solar_data):
    """solar_data DataFrame을 PostgreSQL에 삽입"""
    insert_sql = df_to_insert_sql(
        solar_data, "solar_data", [timestamp_col, "solar_id", "generation"]
    )
    try:
        with db_connect.cursor() as cur:
            cur.execute(insert_sql)
            db_connect.commit()
    except Exception as e:
        db_connect.rollback()
        print("[ERROR] SQL 실행 중 오류 발생:", e)
        print("실행 SQL:\n", insert_sql[:500], "...")  # 너무 길면 일부만 출력


def insert_solar_weather(solar_data):
    """solar_weather DataFrame을 PostgreSQL에 삽입"""
    insert_sql = df_to_insert_sql(
        solar_data,
        "solar_weather",
        [
            timestamp_col,
            "solar_id",
            "temperature",
            "humidity",
            "wind_speed",
            "wind_direction",
            "cloud_cover",
        ],
    )
    try:
        with db_connect.cursor() as cur:
            cur.execute(insert_sql)
            db_connect.commit()
    except Exception as e:
        db_connect.rollback()
        print("[ERROR] SQL 실행 중 오류 발생:", e)
        print("실행 SQL:\n", insert_sql[:500], "...")  # 너무 길면 일부만 출력


def drop_table_if_exists(table_name):
    """테이블이 존재하면 삭제"""
    with db_connect.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name};")
        db_connect.commit()


def get_sampled_df(df, id_col, timestamp_col):

    current_hour = datetime.now().hour
    now_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:00:00")

    # 모든 solar_id 확인
    data_ids = df[id_col].unique()
    sampled_rows = []
    for d_id in data_ids:
        hourly_data = df[
            (pd.to_datetime(df[timestamp_col]).dt.hour == current_hour)
            & (df[id_col] == d_id)
        ]
        if not hourly_data.empty:
            sample = hourly_data.sample(1).copy()
            sample[timestamp_col] = now_timestamp
            sampled_rows.append(sample)
        else:
            print(
                f"⚠️ {id_col}={d_id} 에 대해 현재 시간({current_hour})에 해당하는 데이터 없음"
            )
    # 최종 삽입할 DataFrame
    sampled_df = pd.concat(sampled_rows, ignore_index=True)
    return sampled_df


def sleep_until_next_hour():
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    time_to_wait = (next_hour - now).total_seconds()
    print(f"⏱️ Waiting {time_to_wait:.1f} seconds until {next_hour}")
    time.sleep(time_to_wait)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--db-host", dest="db_host", type=str, default="localhost")
    parser.add_argument(
        "--solar-path", dest="solar_path", type=str, default="/usr/app/solar_data.csv"
    )
    parser.add_argument(
        "--load-path", dest="load_path", type=str, default="/usr/app/load_data.csv"
    )

    args = parser.parse_args()

    print("DataBase Host", args.db_host)
    print("Solar CSV:", args.solar_path)
    print("Load CSV:", args.load_path)

    db_connect = psycopg2.connect(
        user="admin",
        password="1234",
        host=args.db_host,
        port=5432,
        database="machinedb",
    )

    while True:
        timestamp_col = "timestamp"

        current_hour = datetime.now().hour
        # Load data
        load_data = pd.read_csv(args.load_path)
        load_id_col = "load_id"
        load_data[timestamp_col] = pd.to_datetime(load_data[timestamp_col])

        # Solar data
        solar_data = pd.read_csv(args.solar_path)
        solar_id_col = "solar_id"
        solar_data[timestamp_col] = pd.to_datetime(solar_data[timestamp_col])

        # Insert data into PostgreSQL
        load_sampled_data = get_sampled_df(load_data, load_id_col, timestamp_col)
        insert_load(load_sampled_data)
        insert_load_weather(load_sampled_data)

        solar_sampled_data = get_sampled_df(solar_data, solar_id_col, timestamp_col)
        insert_solar(solar_sampled_data)
        insert_solar_weather(solar_sampled_data)

        sleep_until_next_hour()

    # with db_connect.cursor() as cur:
    #     # cur.execute("SET TIME ZONE 'Asia/Seoul';")
    #     db_connect.commit()
