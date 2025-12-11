import xarray as xr
import cfgrib
import pandas as pd
import argparse
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data", required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  ds=xr.open_dataset(args.data, engine='cfgrib')
  df=ds.to_dataframe().reset_index()
  df.to_csv(args.output, index=False)