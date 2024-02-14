import requests
import pandas as pd
import argparse
import os

def main():
    """
    Python script which sends an audio file to a transcription web service and passes the returned text to the output CSV file.
    If no output csv file is provided, the generated text is appended to the input CSV file
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8001/asr", help="URL to send POST request",
                        type=str)
    parser.add_argument("--csv-input", required=True, help="Path to open CSV file.",
                        type=str)
    parser.add_argument("--csv-output", default=None, help="Path to store CSV file. If not provided, 'generated_text' column is added to input csv by default.",
                        type=str)
    parser.add_argument("--directory", required=True, help="Directory path containing all audio files for transcription",
                        type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_input)

    gentext_dict = {}

    for filename in df['filename'].tolist():
        filepath = os.path.join(args.directory, filename)

        with open(filepath, 'rb') as soundfile:
            response = requests.post(args.url, files={"file": soundfile})
            rjson = response.json()
            gentext_dict[filename] = rjson["transcription"]

    df["generated_text"] = df["filename"].map(gentext_dict)
    df["generated_text"] = df["generated_text"].astype(str)

    if args.csv_output is None:
        df.to_csv(args.csv_input, index=False)
    else:
        df.to_csv(args.csv_output, index=False)

if __name__ == "__main__":
    main()
