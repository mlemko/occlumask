import occlumask
import sys
import glob
import os
import json
import shutil
from datetime import datetime

# For now, nothing is done to verify outputs, since test results are evaluated qualitatively.
# Once a parseable and quantitative output format is worked out, test cases will be paired with expected value ranges.

OUTPUT_ROOT = "test-output"
TEST_ROOT = "test-dialogue"

shutil.make_archive(datetime.now().isoformat(), "zip", base_dir=OUTPUT_ROOT)

def do_tests_with_model(model_path):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(os.path.join(OUTPUT_ROOT, "prompt.txt"), "w+") as promt_file:
        promt_file.write("User Prompt:\n")
        promt_file.write(occlumask.user_prompt)
        
    for file in glob.iglob("./**/*.txt", root_dir=TEST_ROOT, recursive=True):
        print(f"Testing: {file}")
        result_text = os.path.join(OUTPUT_ROOT ,file[2:])
        result_json = os.path.join(OUTPUT_ROOT, f"{file[2:]}.json")
        source_file = os.path.join(TEST_ROOT, f"{file[2:]}")
        
        text, resp = occlumask.main(model_path, [source_file])
        
        os.makedirs(os.path.split(result_text)[0], exist_ok=True)
        with open(result_text, "w+") as text_outf:
            text_outf.write(text)
            text_outf.write(resp["choices"][0]["text"])
        
        with open(result_json, "w+") as json_outf:
            json.dump(resp, json_outf, indent=4)
        
        print()
        
    shutil.make_archive(datetime.now().isoformat(), "zip", base_dir=OUTPUT_ROOT)
            

if __name__ == "__main__":
    do_tests_with_model(sys.argv[1])
