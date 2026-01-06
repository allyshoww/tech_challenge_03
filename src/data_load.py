import argparse, logging, pandas as pd, requests, os
from io import StringIO
logging.basicConfig(level=logging.INFO); log = logging.getLogger(__name__)

def main(url, output):
    log.info("1. Baixando dataset...")
    df = pd.read_csv(StringIO(requests.get(url).text))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    log.info(f"Dataset raw: {output}, shape={df.shape}, cols={list(df.columns)}")
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--url',default='https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Airline%20Delay.csv'); p.add_argument('--output','-o',required=True)
    main(**vars(p.parse_args()))
