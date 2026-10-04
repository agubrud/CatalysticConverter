import argparse
import yaml
import sys
import os

def process_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_yaml', type=str, required=True)
    args = parser.parse_args()

    return args

def plot_data(input_data: dict):
    # use matplotlib and pandas to create plots
    import matplotlib.pyplot as plt
    import pandas as pd

     # don't display the plot, only save it later
    fig = plt.figure(figsize=(10, 4))

    # for each CSV file in the input data, load it into a df
    for file_name, file_info in input_data.items():
        csv_path = file_info['path']
        df = pd.read_csv(csv_path, sep=';')

        # smooth the speed data
        df['speed'] = df['speed'].rolling(window=10, center=True).mean()

        # normalize the speed data x axis so each entry takes up the whole width
        df['frame_number'] = (df['frame_number'] - df['frame_number'].min()) / (df['frame_number'].max() - df['frame_number'].min())

        plt.plot(df['frame_number'], df['speed'], label=f'{file_name}')
        #plt.plot(df['frame_number'], df['gforce_cY'], label='G-Force Y')

    plt.xlabel('Normalized Lap Time')
    plt.ylabel('Speed (MPH)')
    plt.title('Speed Data Over Time')
    plt.legend()
    
    #plt.show()
    # save to a png
    plt.savefig("results/mult.png")
    plt.close(fig)


if __name__ == "__main__":
    args = process_cli_args()

    input_data = yaml.safe_load(open(args.input_yaml, 'r'))

    plot_data(input_data)