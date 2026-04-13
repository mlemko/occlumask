from argparse import Namespace
import occlumask
import sys
import glob
import os
import json
import shutil
import random
from datetime import datetime

# Results are evaluated binarily, based on whether the LLM matches the classification given by humans.

OUTPUT_ROOT = "test-output"
TEST_ROOT = "test-dialogue"
NUM_OF_SEEDS = 10

def do_tests_with_model(model_path):
    test_time = datetime.now()
    random.seed()
    used_seeds = set[int]()
    summary = {"fails": [], "success": [], "total_t": 0, "total_f": 0, "success_t": 0}
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
        
    for file in glob.iglob("./**/*.txt", root_dir=TEST_ROOT, recursive=True):
        summary["total_f"] += 1
        used_seeds.clear()
        file_summary = {"correct": [], "incorrect": [], "error": []}
        print(f"Testing: {file}")
        result_folder = os.path.join(OUTPUT_ROOT, file[2:-4])
        source_file = os.path.join(TEST_ROOT, f"{file[2:]}")
        expected_res = "positive" in os.path.split(os.path.dirname(source_file))[1]
        os.makedirs(result_folder, exist_ok=True)
        
        for i in range(NUM_OF_SEEDS):
            seed = random.randint(0, sys.maxsize)
            while seed in used_seeds:
                seed = random.randint(0, sys.maxsize)
            used_seeds.add(seed)
            print(f"Using seed '{seed}'")
            summary["total_t"] += 1
            text, resp = occlumask.main(Namespace(model_path=model_path, input_file=source_file, pretty_print=False, seed=seed))
            resp_text = resp["choices"][0]["text"]
            neg_res = resp_text.rfind("RESPONSE: NO\n") != -1
            pos_res = resp_text.rfind("RESPONSE: YES\n") != -1
            test_name = f"{seed}"
            # TODO: refactor this tree please. its ugly as hell
            if neg_res ^ pos_res:
                if neg_res:
                    test_name = "NEG-" + test_name
                    if not expected_res:
                        test_name = "CORRECT-" + test_name
                        file_summary["correct"].append(test_name)
                    else:
                        test_name = "INCORRECT-" + test_name
                        file_summary["incorrect"].append(test_name)
                else:
                    test_name = "POS-" + test_name
                    if expected_res:
                        test_name = "CORRECT-" + test_name
                        file_summary["correct"].append(test_name)
                    else:
                        test_name = "INCORRECT-" + test_name
                        file_summary["incorrect"].append(test_name)
            else:
                test_name = "ERR-" + test_name
                file_summary["error"].append(test_name)
            
            result_file = os.path.join(result_folder,test_name)
            with open(result_file + ".txt", "w+") as text_outf:
                text_outf.write(text)
                text_outf.write(resp["choices"][0]["text"])
            
            with open(result_file + ".json", "w+") as json_outf:
                json.dump(resp, json_outf, indent=4)
        
        summary["success_t"] += len(file_summary["correct"])
        print("\nPROMPT SUMMARY:\n")
        print(f"{len(file_summary['correct'])} seeds gave a correct answer.")
        print(f"{len(file_summary['incorrect'])} seeds gave an incorrect answer.")
        
        with open(os.path.join(result_folder, "summary.txt"), "+w") as psum_outf:
            psum_outf.write("PROMPT SUMMARY:\n\n")
            psum_outf.write(f"Prompt name: {os.path.split(result_folder)[1]}\n")
            psum_outf.write(f"Number of attempts: {NUM_OF_SEEDS}\n")
            psum_outf.write(f"{len(file_summary['correct'])} tests gave the correct answer.\n")
            psum_outf.write(f"{len(file_summary['incorrect'])} tests gave the incorrect answer{':' if len(file_summary['incorrect']) else '.'}\n")
            for test in file_summary["incorrect"]:
                psum_outf.write(f"{test}\n")
            psum_outf.write(f"\n{len(file_summary['error'])} did not give a valid answer{':' if len(file_summary['error']) else '.'}\n")
            for test in file_summary["error"]:
                psum_outf.write(f"{test}\n")
        
        if len(file_summary["error"]) or len(file_summary["incorrect"]):
            summary["fails"].append(f"{result_folder}")
        else:
            summary["success"].append(f"{result_folder}")
        
    
    # write whole test summary
    with open(os.path.join(OUTPUT_ROOT, "summary.txt"), "w+") as sum_outf:
        sum_outf.write(f"TEST SUMMARY - {test_time.isoformat()}\n\n")
        sum_outf.write(f"Model used: {os.path.split(model_path)[1]}\n")
        sum_outf.write(f"Number of attempts per prompt: {NUM_OF_SEEDS}\n")
        sum_outf.write("User Prompt:\n")
        sum_outf.write(occlumask.user_prompt)
        
        sum_outf.write("\n\n")
        sum_outf.write(f"{summary['total_f']} test prompts found.\n")
        sum_outf.write(f"{len(summary['success'])} prompts succeeded without fail.\n")
        sum_outf.write(f"{len(summary['fails'])} prompts had a failed test{':' if len(summary['fails']) else '.'}\n")
        for prompt in summary['fails']:
            sum_outf.write(os.path.relpath(prompt, TEST_ROOT) + "\n")
        
        sum_outf.write("\n\n")
        sum_outf.write(f"{summary['total_t']} tests ran.\n")
        sum_outf.write(f"{summary['success_t'] * 100.00 / summary['total_t']}% of tests succeeded.\n")
        sum_outf.write(f"{summary['success_t']} tests succeeded.\n")
        sum_outf.write(f"{summary['total_t'] - summary['success_t']} tests failed.\n")
        
    shutil.make_archive(test_time.isoformat(), "zip", base_dir=OUTPUT_ROOT)

if __name__ == "__main__":
    do_tests_with_model(sys.argv[1])
