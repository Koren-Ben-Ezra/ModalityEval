# # import pandas as pd
# # import json

# # metadata = {
# #     "test name": "basic test",
# #     "Model": "Llama",
# #     "Dataset": "GSM8k",
# #     "Save Predictions": True,
# # }

# # columns = ["Text Filter", "Text Correct", "Image Filter", "Image Correct", "Total Samples"]
# # summary_df = pd.DataFrame(columns=columns)

# # # Save the DataFrame as CSV
# # df_filename = metadata.get("test name", "summary").replace(" ", "_") + ".csv"
# # summary_df.to_csv(df_filename, index=False)

# # # Save the metadata as a JSON file
# # metadata_filename = df_filename.replace(".csv", "_metadata.json")
# # with open(metadata_filename, "w") as f:
# #     json.dump(metadata, f, indent=4)

# from decimal import Decimal, InvalidOperation

# def clean_str_number(s: str) -> str:
#     if not s:
#         return s
    
#     try:
#         d = Decimal(s)
#     except InvalidOperation:
#         return s
    
#     d_normalized = d.normalize()
#     s_formatted = format(d_normalized, 'f')
    
#     if '.' in s_formatted:
#         s_formatted = s_formatted.rstrip('0').rstrip('.')
    
#     return s_formatted


#     # Example usage
# test_cases = [
#     ("14.00", "14"),
#     ("14.5", "14.5"),
#     ("0.00", "0"),
#     ("123.45000", "123.45"),
#     ("100", "100"),
#     ("0", "0"),
#     ("0.10", "0.1"),
#     ("10.000", "10"),
#     ("10.01", "10.01"),
#     ("", ""),  # Edge case: empty string
#     ("abc", "abc"),  # Edge case: non-numeric string
# ]

# # Run tests
# for input_str, expected_output in test_cases:
#     result = clean_str_number(input_str)
#     print(f"Input: '{input_str}' | Expected: '{expected_output}' | Result: '{result}' | Pass: {result == expected_output}'")


import re

text = """
[QUESTION] Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

**Step 1: Calculate the total number of eggs laid by Janet's ducks per day.**

Janet's ducks lay 16 eggs per day.

**Step 2: Calculate the number of eggs Janet eats for breakfast and bakes into muffins per day.**

She eats 3 eggs for breakfast and bakes 4 eggs into muffins every day.

Total eggs used for breakfast and baking = 3 + 4 = 7 eggs per day.

**Step 3: Calculate the number of eggs remaining to be sold at the farmers' market per day.**

Remaining eggs = Total eggs - Eggs used for breakfast and baking
= 16 - 7
= 9 eggs per day.

**Step 4: Calculate the total amount Janet makes from selling the remaining eggs at the farmers' market per day.**

Price per egg = $2
Total amount made = Price per egg * Number of eggs sold
= $2 * 9
= $18 per day.

**Answer:** $18<|eot_id|>"""


import unittest
import re

def extract_answer(text: str, token: str = "<|eot_id|>") -> str:
    pattern = r"([-+]?\d+(?:\.\d+)?)(?:\s*" + re.escape(token) + ")"
    match = re.search(pattern, text)
    if match:
        return str(match.group(1))
    else:
        return None

class TestExtractAnswer(unittest.TestCase):

    def test_positive_integer(self):
        self.assertEqual(extract_answer("answer: 18 <|eot_id|>"), "18")

    def test_negative_integer(self):
        self.assertEqual(extract_answer("answer: -42 <|eot_id|>"), "-42")

    def test_positive_float(self):
        self.assertEqual(extract_answer("answer: 3.14 <|eot_id|>"), "3.14")

    def test_negative_float(self):
        self.assertEqual(extract_answer("answer: -0.001 <|eot_id|>"), "-0.001")

    def test_number_with_extra_space(self):
        self.assertEqual(extract_answer("answer: 99   <|eot_id|>"), "99")

    def test_no_number(self):
        self.assertIsNone(extract_answer("answer: hello <|eot_id|>"))

    def test_token_not_found(self):
        self.assertIsNone(extract_answer("answer: 18 <END>"))

    def test_real(self):
        self.assertEqual(extract_answer(text), "18")
if __name__ == "__main__":
    unittest.main()
