import json
import os
import urllib
import urllib.request


def download_and_load_file(file_path, url):
    """
    Downloads a file from the specified URL and loads it into memory.

    Args:
        file_path (str): The local path where the file should be saved or loaded.
        url (str): The URL of the file to download.

    Returns:
        dict: The data loaded from the file.
    """

    if not os.path.exists(file_path):

        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")

        with open(file_path, "w", encoding="utf-8") as file:
            text_data = file.write(text_data)

    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r") as file:
        data = json.load(file)

    return data


file_path = "fine-tuning-instruct/data/instruction-data.json"

url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)

# print("Number of entries:", len(data))


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
# print(model_input + desired_response)
# print("\n")

model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"
# print(model_input + desired_response)
