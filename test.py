# import pandas as pd
# import json

# metadata = {
#     "test name": "basic test",
#     "Model": "Llama",
#     "Dataset": "GSM8k",
#     "Save Predictions": True,
# }

# columns = ["Text Filter", "Text Correct", "Image Filter", "Image Correct", "Total Samples"]
# summary_df = pd.DataFrame(columns=columns)

# # Save the DataFrame as CSV
# df_filename = metadata.get("test name", "summary").replace(" ", "_") + ".csv"
# summary_df.to_csv(df_filename, index=False)

# # Save the metadata as a JSON file
# metadata_filename = df_filename.replace(".csv", "_metadata.json")
# with open(metadata_filename, "w") as f:
#     json.dump(metadata, f, indent=4)

from decimal import Decimal, InvalidOperation

def clean_str_number(s: str) -> str:
    if not s:
        return s
    
    try:
        d = Decimal(s)
    except InvalidOperation:
        return s
    
    d_normalized = d.normalize()
    s_formatted = format(d_normalized, 'f')
    
    if '.' in s_formatted:
        s_formatted = s_formatted.rstrip('0').rstrip('.')
    
    return s_formatted


    # Example usage
test_cases = [
    ("14.00", "14"),
    ("14.5", "14.5"),
    ("0.00", "0"),
    ("123.45000", "123.45"),
    ("100", "100"),
    ("0", "0"),
    ("0.10", "0.1"),
    ("10.000", "10"),
    ("10.01", "10.01"),
    ("", ""),  # Edge case: empty string
    ("abc", "abc"),  # Edge case: non-numeric string
]

# Run tests
for input_str, expected_output in test_cases:
    result = clean_str_number(input_str)
    print(f"Input: '{input_str}' | Expected: '{expected_output}' | Result: '{result}' | Pass: {result == expected_output}'")
