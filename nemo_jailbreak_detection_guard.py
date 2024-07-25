from typing import List
import time
import statistics

from nemoguardrails import LLMRails, RailsConfig

OUTFILE = "nemo_jailbreak_guard_gpt4_selfcheckinput.txt"


def evaluate_nemo_guard_on_dataset(rails, test_prompts: List[str], verbose: bool, expect_pass: bool, outfilepath: str):
    nemo_cannot_respond = "Julia I'm sorry, I can't respond to that."  # Defined in config/flows.co
    num_failed_guard = 0
    num_passed_guard = 0
    latency_measurements = []
    for i, prompt in enumerate(test_prompts):
        try:
            start_time = time.perf_counter()
            response = rails.generate(messages=[{
                "role": "user",
                "content": prompt
            }])
            latency = time.perf_counter() - start_time
            latency_measurements.append(latency)
            
            if nemo_cannot_respond in response["content"]:
                num_failed_guard += 1
                with open(OUTFILE, "a") as f:
                    f.write(f"\n\nPROMPT FAILS")
            else:
                num_passed_guard += 1
                with open(OUTFILE, "a") as f:
                    f.write(f"\n\nPROMPT PASSES")
            
            total = num_passed_guard + num_failed_guard
            with open(OUTFILE, "a") as f:
                f.write(f"\n\nprompt:\n{prompt}")
                f.write(f"\n\nresponse:\n{response}")
                f.write(f"\n\n{100 * num_failed_guard / total:.2f}% of {total} prompts failed the NeMo Jailbreak guard.")
                f.write(f"\n\n{100 * num_passed_guard / total:.2f}% of {total} prompts passed the NeMo Jailbreak guard.")
        except Exception as e:
            with open(OUTFILE, "a") as f:
                f.write(f"\n\nerror on prompt:\n{str(e)}")

    return num_passed_guard, num_failed_guard, latency_measurements


def benchmark_nemo(jailbreak_test_prompts, vanilla_prompts):
    # NeMo Guard on Jailbreak dataset
    config = RailsConfig.from_path("/Users/juliagomes/validator-template/validator/config/")
    rails = LLMRails(config)
    with open(OUTFILE, "a") as f:
        f.write(f"\nEvaluate the NeMo Jailbreak Guard against {len(jailbreak_test_prompts)} examples")
    num_passed_nemo, num_failed_nemo, latency_measurements = evaluate_nemo_guard_on_dataset(
        rails=rails,
        test_prompts=jailbreak_test_prompts,
        verbose=True,
        expect_pass=False,
        outfilepath="/Users/juliagomes/validator-template/validator/nemo_false_negatives.csv")
    with open(OUTFILE, "a") as f:
        f.write(f"\n{num_failed_nemo} True Positives")
        f.write(f"\n{num_passed_nemo} False Negatives")
        f.write(f"\n{statistics.median(latency_measurements)} median latency\n{statistics.mean(latency_measurements)} mean latency\n{max(latency_measurements)} max latency")

    # NeMo Guard on Vanilla dataset
    with open(OUTFILE, "a") as f:
        f.write(f"\nEvaluate the NeMo Jailbreak Guard against {len(vanilla_prompts)} examples")
    num_passed_nemo, num_failed_nemo, latency_measurements = evaluate_nemo_guard_on_dataset(
        rails=rails,
        test_prompts=vanilla_prompts,
        verbose=True,
        expect_pass=False,
        outfilepath="/Users/juliagomes/validator-template/validator/nemo_false_positives.csv")
    with open(OUTFILE, "a") as f:
        f.write(f"\n{num_passed_nemo} True Negatives")
        f.write(f"\n{num_failed_nemo} False Positives")
        f.write(f"\n{statistics.median(latency_measurements)} median latency\n{statistics.mean(latency_measurements)} mean latency\n{max(latency_measurements)} max latency")
