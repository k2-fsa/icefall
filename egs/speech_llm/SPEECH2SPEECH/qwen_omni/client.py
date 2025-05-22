# client.py
import argparse
import json
import os

import requests
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Speech-to-Text Client")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost",
        help="URL of the FastAPI server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port of the FastAPI server",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="hlt-lab/voicebench",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--subset-name",
        type=str,
        default="commoneval",  # Adjust as needed
        help="Dataset subset name",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default=None,  # Adjust as needed
        help="Dataset split name",
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, help="Directory to save results"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = os.path.join(
        args.output_dir,
        f"{args.subset_name}-{args.split_name}.jsonl",
    )
    server_decode_url = f"{args.server_url}:{args.port}/decode"

    print("Loading dataset...")
    if args.subset_name != "mmsu":
        dataset = load_dataset(
            args.dataset_name,
            args.subset_name,
            split=args.split_name,
            trust_remote_code=True,
        )
    else:
        # load all splits and concatenate them
        dataset = load_dataset(
            args.dataset_name,
            args.subset_name,
            trust_remote_code=True,
        )
        dataset = concatenate_datasets([dataset[subset] for subset in dataset])

    print(f"Dataset loaded with {len(dataset)} samples.")
    print(f"Sending requests to {server_decode_url}...")
    print(f"Saving results to {output_filename}")

    with open(output_filename, "w", encoding="utf-8") as outfile:
        # Iterate directly over the dataset
        progress_bar = tqdm(dataset, desc="Processing", unit="samples")
        for item in progress_bar:

            audio_info = item.get("audio")
            assert (
                audio_info["sampling_rate"] == 16000
            ), f"Sampling rate is {audio_info['sampling_rate']}, not 16khz"

            # Prepare data for JSON serialization and server request
            audio_array = audio_info["array"].tolist()  # Convert numpy array to list
            result_dict = {}
            for key in item.keys():
                if key != "audio":
                    # Ensure other fields are JSON serializable
                    try:
                        # Attempt to serialize to catch issues early (optional)
                        json.dumps(item[key])
                        result_dict[key] = item[key]
                    except (TypeError, OverflowError):
                        print(
                            f"Warning: Converting non-serializable key '{key}' to string."
                        )
                        result_dict[key] = str(
                            item[key]
                        )  # Convert problematic types to string

            payload = {
                "audio": audio_array,
                "sampling_rate": 16000,
            }

            try:
                response = requests.post(server_decode_url, json=payload, timeout=60)
                response.raise_for_status()
                server_response = response.json()
                decoded_text = server_response.get("text", "")

                # Add the response to the result dictionary
                result_dict["response"] = decoded_text
                print(result_dict)
                # Write result to JSONL file
                json.dump(result_dict, outfile, ensure_ascii=False)
                outfile.write("\n")

            except requests.exceptions.RequestException as e:
                print(f"\nError sending request for an item: {e}")
                error_entry = result_dict  # Use the data prepared so far
                error_entry["error"] = str(e)
                error_entry["response"] = ""
                json.dump(error_entry, outfile, ensure_ascii=False)
                outfile.write("\n")
            except json.JSONDecodeError:
                print("\nError decoding server response for an item.")
                error_entry = result_dict
                error_entry["error"] = "Invalid JSON response from server"
                error_entry["response"] = ""
                json.dump(error_entry, outfile, ensure_ascii=False)
                outfile.write("\n")
            except Exception as e:
                print(f"\nUnexpected error processing an item: {e}")
                error_entry = result_dict
                error_entry["error"] = f"Unexpected error: {str(e)}"
                error_entry["response"] = ""
                json.dump(error_entry, outfile, ensure_ascii=False)
                outfile.write("\n")

            # Progress bar updates automatically by iterating over tqdm(dataset)

        # No need to close progress_bar explicitly when iterating directly

    print("Processing finished.")


if __name__ == "__main__":
    main()
