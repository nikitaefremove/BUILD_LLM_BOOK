from utils.data_processing.downloading_dataset import data


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


# model_input = format_input(data[50])
# desired_response = f"\n\n### Response:\n{data[50]['output']}"
# print(model_input + desired_response)
# print("\n")

# model_input = format_input(data[999])
# desired_response = f"\n\n### Response:\n{data[999]['output']}"
# print(model_input + desired_response)
