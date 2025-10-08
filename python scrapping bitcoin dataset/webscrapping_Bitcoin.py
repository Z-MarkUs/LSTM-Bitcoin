# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:09:54 2024

@author: loren
"""

import requests
import time
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from itertools import permutations
from typing import Callable

def to_snake_case(text: str) -> str:
    replacements = {' ': '_', '-': '_', '(': '', ')': '', ">": '', ',': ''}
    snake_case_text = text.lower().translate(str.maketrans(replacements)).replace("___", '_').replace("__", '_')
    return snake_case_text

def get_months_since_genesis() -> list[int]:
    genesis = datetime(2009, 1, 1)
    years = np.arange(2009, 2025, step=1)
    months = np.arange(1, 13, step=1)
    years, months = np.meshgrid(years, months)
    perms = np.column_stack((years.ravel(), months.ravel()))
    order = np.lexsort((perms[:, 1], perms[:, 0]))
    perms = perms[order]
    ts = [int(datetime(r[0], r[1], 1).timestamp()) for r in perms if datetime(r[0], r[1], 1).date() < datetime.now().date()]
    return ts

def blockchain_dot_com_get_request(endpoint: str, month: int) -> tuple[dict, str, str] | None:
    base_url= "https://api.blockchain.info/charts/"
    params = {"timespan": "5weeks", "format": "json", "start": month}
    for retry in range(5):
        try:
            r = requests.get(base_url + endpoint, params=params)
            data = r.json()["values"]
            desc = r.json()["description"]
            name = to_snake_case(r.json()["name"])
            return data, desc, name
        except Exception as e:
            time.sleep(random.randint(1, retry + 1))
            if retry == 4:
                msg = f"""Blockchain.com API endpoint seems broken...\n
                baseurl: {base_url}\n
                endpoint: {endpoint}\n
                params: {params}\n
                month: {datetime.from_timestamp(month)}\n
                exception: {type(e)}\n
                exception_args: {e.args}"""
                print(msg)
    else:
        return None

def get_blockchain_dot_com_endpoint_data(endpoint: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    column_data = []
    for month in get_months_since_genesis():
        response = blockchain_dot_com_get_request(endpoint, month)
        if not response:
            continue
        data, desc, name = response
        column_data += data
    column_df = pd.DataFrame(column_data).drop_duplicates().rename(columns={'x': "datetime", 'y': name})
    column_df["datetime"] = pd.to_datetime(column_df["datetime"], unit='s')
    describe_df = pd.DataFrame({"description": desc}, index=[name])
    return column_df, describe_df
    
def get_all_blockchain_dot_com_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoints = ["mempool-size", "transactions-per-second", "market-cap", "avg-block-size", "market-price",
                 "trade-volume", "avg-confirmation-time", "hash-rate", "difficulty", "miners-revenue", "transaction-fees"]
    timeseries_df = pd.Series([], dtype=int)
    info_df = pd.Series([], dtype=int)
    for e in tqdm(endpoints, desc="Blockchain.com endpoints scraped"):
        column_df, describe_df = get_blockchain_dot_com_endpoint_data(e)
        if timeseries_df.empty:
            timeseries_df = column_df
        else:
            timeseries_df = pd.merge(timeseries_df, column_df, on='datetime', how='outer')
        if info_df.empty:
            info_df = describe_df
        else:
            info_df = pd.concat([info_df, describe_df])
    return timeseries_df, info_df

bdc_timeseries_df, bdc_info_df = get_all_blockchain_dot_com_data()


def clean_blockchain_dot_com_data(ts_df: pd.DataFrame, info_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rename_map = {"market_capitalization": "market_cap_usd", "usd_exchange_trade_volume": "exchange_volume_usd"}
    ts_df = ts_df.rename(columns=rename_map)
    daily_data = (ts_df.sort_values("datetime").groupby(ts_df.datetime.dt.date).median(numeric_only=False)
                  .iloc[:-1].fillna(0).drop("datetime", axis=1))
    half_hourly_data = (ts_df[["datetime", "transaction_rate", "mempool_size", "market_cap_usd"]]
                        .sort_values("datetime").set_index("datetime").resample("30T").median())
    info_data = info_df.rename(index=rename_map)
    info_data.at["average_confirmation_time", "description"] = """The average time taken for a transaction to be combined 
    in a Bitcoin block with other transactions and added to the blockchain."""  # missing from the API inexplicably
    return daily_data, half_hourly_data, info_data

bdc_daily_data, bdc_half_hourly_data, bdc_info_data = clean_blockchain_dot_com_data(bdc_timeseries_df, bdc_info_df)



def lib_post_request(endpoint: str, payload: dict):
    base_url = "https://www.lookintobitcoin.com/django_plotly_dash/app/_ENDPOINT_/_dash-update-component".replace("_ENDPOINT_", endpoint)
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-GB,en;q=0.8",
        "Content-Type": "application/json",
        "User-Agent": """Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1"""
    }
    for retry in range(5):
        try:
            r = requests.post(base_url, headers=headers, json=payload)
            if r.status_code != 200:
                print(r.status_code)
                print(r.text)
                print(base_url)
            return r.json()["response"]["chart"]["figure"]["data"]
        except Exception as e:
            time.sleep(random.randint(1, retry + 1))
            if retry == 4:
                msg = f"""LookintoBitcoin API seems broken...\n baseurl: {base_url}\n exception: {type(e)}\n exception_args: {e.args}"""
                raise Exception(msg)

def scrape_lib_fear_greed(payload: dict) -> pd.DataFrame:  # irregular payload format
    payload["output"] = "..chart.figure...indicator.figure...last_update.children...now.children...now.style.."
    payload["outputs"] = [{"id":"chart","property":"figure"}, {"id":"indicator","property":"figure"},
                          {"id":"last_update","property":"children"}, {"id":"now","property":"children"},
                          {"id":"now","property":"style"}]
    payload["inputs"][0]["values"] = "/charts/bitcoin-fear-and-greed-index/"
    request_data = lib_post_request("fear_and_greed", payload)[1]["customdata"]
    df = pd.DataFrame([[c[i] for i in (0, 2, 3)] for c in request_data],
                      columns=["datetime", "fear_greed_value", "fear_greed_category"])
    return df

def scrape_lib_mvrv(payload: dict) -> pd.DataFrame:
    payload["inputs"][0]["value"] = "/charts/mvrv-zscore/"
    request_data = lib_post_request("mvrv_zscore", payload)
    mcap_data = [data for data in request_data if data["name"] == "Market Cap"][0]
    rcap_data = [data for data in request_data if data["name"] == "Realized Cap"][0]
    column_values = np.array([mcap_data['y'], rcap_data['y']]).T
    index_values = mcap_data["x"][:len(mcap_data['y']) - len(mcap_data['x'])]
    df = pd.DataFrame(column_values, columns=["market_cap_usd", "realised_cap_usd"], index=index_values).reset_index(names="datetime")
    return df

def scrape_lib_nupl(payload: dict) -> pd.DataFrame:  # inconsistent '_' versus '-' usage in the payload["inputs"][x]["value"], see mvrv vs. nupl
    payload["inputs"][0]["value"] = "/charts/unrealised_profit_loss/"
    request_data = lib_post_request("unrealised_profit_loss", payload)
    price_data = [data for data in request_data if data["name"] == "BTC Price"][0]
    nupl_data = [data for data in request_data if data["name"] == "Net Unrealised Profit / Loss (NUPL)"][0]
    column_values = np.array([price_data["y"], nupl_data["y"]]).T
    index_values = price_data["x"][:len(price_data["y"]) - len(price_data["x"])]
    df = pd.DataFrame(column_values, columns=["btc_price_usd", "nupl"], index=index_values).reset_index(names="datetime")
    return df

def scrape_lib_cdd(payload: dict) -> pd.DataFrame:  # where does bdd come from??? typo??? why does it work???
    payload["inputs"][0]["value"] = "/charts/coin-days-destroyed-cdd/"
    request_data = lib_post_request("bdd", payload)
    cdd_data = [data for data in request_data if data["name"] == "CDD (raw data)"][0]
    index_values = cdd_data['x'][:len(cdd_data['y']) - len(cdd_data['x'])]
    df = pd.DataFrame(cdd_data['y'], columns=["coin_days_destroyed"], index=index_values).reset_index(names="datetime")
    payload["inputs"][0]["value"] = "/charts/bitcoin-funding-rates/"
    return df

def scrape_lib_aa(payload: dict) -> pd.DataFrame:
    payload["output"] = "..chart.figure...scale_y2.disabled...scale_y2.value...scale_y1.disabled...scale_y1.value.."
    payload["outputs"] = [{"id":"chart","property":"figure"},{"id":"scale_y2","property":"disabled"},
                          {"id":"scale_y2","property":"value"},{"id":"scale_y1","property":"disabled"},
                          {"id":"scale_y1","property":"value"}]
    payload["inputs"] = [{"id":"url","property":"pathname","value":"/charts/bitcoin-active-addresses/"},
                         {"id":"scale_y2","property":"value","value":"log"},
                         {"id":"scale_y1","property":"value","value":"linear"}]
    request_data = lib_post_request("active_addresses", payload)
    aa_data = [data for data in request_data if data["name"] == "Active Addresses Daily Values"][0]
    index_values = aa_data['x'][:len(aa_data['y']) - len(aa_data['x'])]
    df = pd.DataFrame(aa_data['y'], columns=["active_addresses"], index=index_values).reset_index(names="datetime")
    return df


def scrape_lib_fr(payload: dict) -> pd.DataFrame:
    payload["output"] = "..chart.figure...exchange.options...resolution.disabled...resolution.value.."
    payload["outputs"] = [{"id":"chart","property":"figure"},{"id":"exchange","property":"options"},{"id":"resolution","property":"disabled"},{"id":"resolution","property":"value"}]
    payload["inputs"] =  [{"id":"url","property":"pathname","value":"/charts/bitcoin-funding-rates/"},{"id":"currency","property":"value","value":"funding_rate_usd"},{"id":"exchange","property":"value","value":"average"},{"id":"resolution","property":"value","value":"1h"}]
    request_data = lib_post_request("funding_rates", payload)
    if request_data[1]["name"] == 'Funding Rate':
       index_values = request_data[1]['x'][:len(request_data[1]['y'])]
       df = pd.DataFrame(request_data[1]['y'], columns=["funding_rates"], index=index_values).reset_index(names="datetime")
    if request_data[0]["name"] == 'BTC Price':
       df['BTC_price'] = request_data[0]['y']
    df.to_csv("look_into_bitcoin_funding_rates.csv")
    return df

def scrape_lib_lightning_capacity(payload: dict) -> pd.DataFrame:  # why is the data returned without a nametag???
    payload["inputs"][0]["value"] = "/charts/lightning-network-capacity/"
    request_data = lib_post_request("lightning_capacity",payload)
    cap_data = request_data[0]
    index_values = cap_data['x'][:len(cap_data['y']) - len(cap_data['x'])]
    df = pd.DataFrame(cap_data['y'], columns=["lightning_capacity_usd"], index=index_values).reset_index(names="datetime")
    return df

def scrape_lib_lightning_nodes(payload: dict) -> pd.DataFrame:  # still no nametag and now x, y are the same length???
    payload["inputs"][0]["value"] = "/charts/lightning-network-nodes/"
    request_data = lib_post_request("lightning_nodes", payload)
    node_data = request_data[0]
    index_values = node_data["x"]
    df = pd.DataFrame(node_data["y"], columns=["lightning_nodes"], index=index_values).reset_index(names="datetime")
    return df

def scrape_lib_hodl(payload: dict) -> pd.DataFrame:  # wow they actually made a good backend for one of these things
    payload["inputs"][0]["value"] = "/charts/hodl-waves/"
    request_data = lib_post_request("hodl_waves", payload)
    col_names = [to_snake_case(d["name"]) for d in request_data]
    col_values = np.array([d['y'] for d in request_data]).T
    index_values = [d['x'] for d in request_data][0]
    df = pd.DataFrame(col_values, columns=col_names, index=index_values).reset_index(names="datetime")
    return df

def scrape_lib_rcap_hodl(payload: dict) -> pd.DataFrame:
    payload["inputs"][0]["value"] = "/charts/realized-cap-hodl-waves/"
    request_data = lib_post_request("rcap_hodl_waves", payload)
    col_names = [to_snake_case(d["name"]) for d in request_data]
    col_values = np.array([d['y'] for d in request_data]).T
    index_values = [d['x'] for d in request_data][0]
    df = pd.DataFrame(col_values, columns=col_names, index=index_values).reset_index(names="datetime")
    return df

def scrape_lib_add_bal(payload: dict) -> pd.DataFrame:  # similar format to hodl data but across many webpages/endpoints
    payload_endpoint_pairs = {"0-01": "001", "0-1": "01", "1": "1", "10": "10", "100": "100", "1000": "1000"}
    final_df = pd.Series([], dtype=int)
    for payload_num, endpoint_num in payload_endpoint_pairs.items():
        payload["inputs"][0]["value"] = f"/charts/addresses-greater-than-{payload_num}-btc/"
        request_data = lib_post_request(f"min_{endpoint_num}_count", payload)
        col_names = [to_snake_case(d["name"]) for d in request_data]
        col_values = np.array([d['y'] for d in request_data]).T
        index_values = [d['x'] for d in request_data][0]
        index_values = index_values[:col_values.shape[0] - len(index_values)]
        df = pd.DataFrame(col_values, columns=col_names, index=index_values).reset_index(names="datetime")
        if final_df.empty:
            final_df = df
        else:
            final_df = pd.merge(df, final_df, on=["datetime", "btc_price"], how="outer")
    return final_df

def add_lib_payload(request: Callable) -> Callable:
    payload = {
        "output": "chart.figure",
        "outputs": {"id": "chart", "property": "figure"},
        "inputs": [{"id": "url", "property": "pathname"}],
        "changedPropIds": ["url.pathname"]
    }
    return request(payload)

def get_lib_daily_data() -> pd.DataFrame:
    daily_data_df = pd.Series([], dtype=int)
    req_functions = [scrape_lib_fear_greed, scrape_lib_mvrv, scrape_lib_nupl, scrape_lib_cdd, scrape_lib_aa, scrape_lib_lightning_capacity, scrape_lib_lightning_nodes, scrape_lib_fr]
    for req in tqdm(req_functions, desc="Lookintobitcoin pages scraped"):
        column_df = add_lib_payload(req)
        column_df["datetime"] = pd.to_datetime(column_df["datetime"])
        if daily_data_df.empty:
            daily_data_df = column_df
        else:
            daily_data_df = pd.merge(daily_data_df, column_df, on="datetime", how="outer")
    return daily_data_df

def get_lib_extra_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hodl_df = add_lib_payload(scrape_lib_hodl)
    rcap_hodl_df = add_lib_payload(scrape_lib_rcap_hodl)
    add_bal_df = add_lib_payload(scrape_lib_add_bal)
    return hodl_df, rcap_hodl_df, add_bal_df

lib_daily_data = get_lib_daily_data()
lib_hodl_data, lib_rcap_hodl_data, lib_add_bal_data = get_lib_extra_data()


def clean_lib_daily_data(lib_df: pd.DataFrame) -> pd.DataFrame:
    newer_data = ["lightning_nodes", "lightning_capacity_usd", "nupl", "fear_greed_category", "fear_greed_value"]
    lib_df = lib_df.sort_values("datetime").iloc[:-1].reset_index(drop=True)
    lib_df["lightning_nodes"] = lib_df["lightning_nodes"].fillna(method="ffill")
    lib_df[newer_data] = lib_df[newer_data].fillna(0)
    lib_df = lib_df.set_index("datetime").iloc[1:-1]
    lib_df["total_supply"] = (lib_df["market_cap_usd"] / lib_df["btc_price_usd"]).astype(int)
    lib_df = lib_df[["btc_price_usd", "total_supply", "market_cap_usd", "realised_cap_usd", "nupl", "coin_days_destroyed", "active_addresses", "fear_greed_value", "fear_greed_category", "lightning_nodes", "lightning_capacity_usd"]]
    return lib_df.rename(columns={"btc_price_usd": "market_price_usd"})

def generate_lib_info_data() -> pd.DataFrame:
    info_dict = {
        "market_price_usd": "Average USD market price across major Bitcoin exchanges.",
        "total_supply": "The total number of Bitcoins in circulation having been mined since Genesis, the first block in the chain.",
        "market_cap_usd": "The total USD value of bitcoin supply in circulation, as calculated by the daily average market price across major exchanges.",
        "realised_cap_usd": "An alternative to market cap, calculated by taking the total sum of the price of each Bitcoin when it was last moved.",
        "nupl": "Net Unrealized Profit/Loss (NUPL) estimates the total paper profit/loss of Bitcoin investors, calculated with the formula: 1 - realised_cap / market_cap",
        "coin_days_destroyed": "Coin Days Destroyed (CDD) takes the number of coins that have moved on-chain at a particular time and multiples that value by the number of days since those coins were last moved.",
        "active_addresses": "The number of addresses on the Bitcoin blockchain that either sent or received transactions.",
        "fear_greed_value": "Proprietary multi-factor Bitcoin market sentiment index produced by lookintobitcoin.com: a value of 0 indicates extreme levels of investor fear and 100 indicates extreme levels of investor greed.",
        "fear_greed_category": "Maps the numerical value of Fear and Greed to a 5-point Likert scale: Extreme Fear, Fear, Neutral, Greed, Extreme Greed.",
        "lightning_nodes": "The total number of Bitcoin Lightning nodes, measured weekly and forward filled for the rest of the week.",
        "lightning_capacity_usd": "This chart shows the cumulative capacity held by all nodes on the Lightning Network in USD."
    }
    info_df = pd.DataFrame(info_dict, index=["description"]).T
    return info_df

def clean_lib_extra_data(extra_df: pd.DataFrame) -> pd.DataFrame:
    extra_df["datetime"] = pd.to_datetime(extra_df["datetime"])
    extra_df = extra_df.set_index("datetime")
    return extra_df.rename(columns={"btc_price": "market_price_usd"})

lib_daily_data = clean_lib_daily_data(lib_daily_data)
lib_info_data = generate_lib_info_data()
lib_hodl_data, lib_rcap_hodl_data, lib_add_bal_data = [clean_lib_extra_data(df) for df in [lib_hodl_data, lib_rcap_hodl_data, lib_add_bal_data]]



warnings.filterwarnings("ignore")

final_dfs = {
    "blockchain_dot_com_daily_data": bdc_daily_data,
    "blockchain_dot_com_half_hourly_data": bdc_half_hourly_data,
    "blockchain_dot_com_column_desc": bdc_info_data,
    "look_into_bitcoin_daily_data": lib_daily_data,
    "look_into_bitcoin_hodl_waves_data": lib_hodl_data,
    "look_into_bitcoin_realised_cap_hodl_waves_data": lib_rcap_hodl_data,
    "look_into_bitcoin_address_balances_data": lib_add_bal_data,
    "look_into_bitcoin_column_desc": lib_info_data
}

def validate_df(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    for nametag in tqdm(dfs):
        if nametag[-12:] == "_column_desc":
            descriptions = dfs[nametag].index
            columns = dfs[f"{nametag[:-12]}_daily_data"].columns
            assert set(columns).issubset(set(descriptions))
        else:
            # Convert index to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(dfs[nametag].index):
                dfs[nametag].index = pd.to_datetime(dfs[nametag].index)

            start_date = dfs[nametag].index[0]
            end_date = dfs[nametag].index[-1]

            try:
                assert start_date >= pd.Timestamp("2009-01-02")
            except AssertionError:
                print(f"Assertion failed for DataFrame start date: {nametag}")
                print(f"Start date in the DataFrame: {start_date}")

            try:
                assert end_date >= (datetime.now() - timedelta(days=2))
            except AssertionError:
                print(f"Assertion failed for DataFrame end date: {nametag}")
                print(f"End date in the DataFrame: {end_date}")

    return dfs


validated_dfs = validate_df(final_dfs)

import os

def save_data(df_list: dict[str, pd.DataFrame], directory: str) -> None:
    for name, df in tqdm(df_list.items(), desc="Files saved"):
        path = os.path.join(directory, f"{name}.csv")
        df.to_csv(path)

# Specify the directory where you want to save the files
output_directory = 'C:\\Users\\loren\\Downloads\\python scrapping bitcoin dataset'

save_data(validated_dfs, output_directory)


"""
#Regarding funding rates:
#https://www.lookintobitcoin.com/charts/bitcoin-funding-rates/
#https://www.youtube.com/watch?v=NYK_1bVoBfU&ab_channel=BhaveshBhatt
#has information where you can get the payload descriptors if you inspect everything, all the variables

import requests
def main():
    global res
    payload = {
        "output": "..chart.figure...exchange.options...resolution.disabled...resolution.value..",
        "outputs": [{"id":"chart","property":"figure"},{"id":"exchange","property":"options"},{"id":"resolution","property":"disabled"},{"id":"resolution","property":"value"}],
        "inputs":  [{"id":"url","property":"pathname","value":"/charts/bitcoin-funding-rates/"},{"id":"currency","property":"value","value":"funding_rate_usd"},{"id":"exchange","property":"value","value":"average"},{"id":"resolution","property":"value","value":"1h"}]
    }

    try:
        result = lib_post_request("funding_rates", payload)
        res = result
        print("Request successful. Data received:")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()


for data in res:
  print(data["name"])
  
print(res[1]["name"])
print(res[1].keys())
print(res[1]['y']) #the funding rates
print(res[1]['x']) #the dates


 """