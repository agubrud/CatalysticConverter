import argparse
import yaml
import sys
import os
# use matplotlib and pandas to create plots
import matplotlib.pyplot as plt
import pandas as pd

def process_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_yaml', type=str, required=True)
    args = parser.parse_args()

    return args

def plot_data(input_data: dict, attribute: str, segment_filter: int = 0):
     # don't display the plot, only save it later
    fig = plt.figure(figsize=(10, 4))

    # for each CSV file in the input data, load it into a df
    for file_name, file_info in input_data.items():
        csv_path = file_info['path']
        df = pd.read_csv(csv_path, sep=';')

        # if a segment filter is provided, only include rows where the segment column matches
        if segment_filter:
            df = df[df['segment'] == segment_filter]

        # smooth the speed data
        df[attribute] = df[attribute].rolling(window=10, center=True).mean()

        # normalize the attribute data x axis so each entry takes up the whole width
        df['frame_number'] = (df['frame_number'] - df['frame_number'].min()) / (df['frame_number'].max() - df['frame_number'].min())

        plt.plot(df['frame_number'], df[attribute], label=f'{file_name}')
        #plt.plot(df['frame_number'], df['gforce_cY'], label='G-Force Y')

    plt.xlabel('Normalized Lap Time')
    plt.ylabel(attribute.title())
    plt.title(f'{attribute.title()} Data Over Time')
    plt.legend()
    
    #plt.show()
    # save to a png
    if segment_filter == 0:
        plt.savefig(f"results/mult_{attribute}.png")
    else:
        plt.savefig(f"results/mult_{attribute}_segment{segment_filter}.png")
    plt.close(fig)


if __name__ == "__main__":
    args = process_cli_args()

    input_data = yaml.safe_load(open(args.input_yaml, 'r'))

    plot_data(input_data, attribute="speed")
    plot_data(input_data, attribute="gforce_cX")
    plot_data(input_data, attribute="gforce_cY")
    plot_data(input_data, attribute="gforce_cMag")

    num_segments = 0
    for file_name, file_info in input_data.items():
        csv_path = file_info['path']
        df = pd.read_csv(csv_path, sep=';')

        # return set of unique values in the segment column
        unique_segments = df['segment'].unique()
        if num_segments != 0 and num_segments != len(unique_segments):
            print(f"Warning: File {file_name} has a different number of segments than expected.")
        num_segments = len(unique_segments)

    for segment in unique_segments:
        plot_data(input_data, attribute="speed", segment_filter=segment)
        plot_data(input_data, attribute="gforce_cX", segment_filter=segment)
        plot_data(input_data, attribute="gforce_cY", segment_filter=segment)
        plot_data(input_data, attribute="gforce_cMag", segment_filter=segment)