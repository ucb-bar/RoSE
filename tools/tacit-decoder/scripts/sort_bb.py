import argparse
import pandas as pd

THRESHOLD = 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sort a csv file by some columns.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input csv file.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    head_df = df.sort_values(by='count', ascending=False)
    head_df.head(100).to_csv(f'{args.input}.head.csv', index=False)

    std_df = df[df['count'] > THRESHOLD]
    std_df = std_df.sort_values(by='netvar', ascending=False)
    std_df.head(1000).to_csv(f'{args.input}.std.csv', index=False)

    mean_df = df[df['count'] > THRESHOLD]
    mean_df = mean_df.sort_values(by='mean', ascending=False)
    mean_df.head(100).to_csv(f'{args.input}.mean.csv', index=False)

    df['total_time'] = df['mean'] * df['count']
    df = df.sort_values(by='total_time', ascending=False)
    df.head(100).to_csv(f'{args.input}.total_time.csv', index=False)
