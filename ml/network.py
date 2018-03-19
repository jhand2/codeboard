#!/usr/bin/python3
from data_utils import load_all_data

if __name__ == "__main__":
    data_cache = load_all_data(45)
    print(data_cache["labels"])
