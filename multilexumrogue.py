import random
from datasets import load_dataset
from rouge_score import rouge_scorer
import csv
import time
import gc  # Import garbage collection module

# Function to process a subset of instances
def process_instances(start_idx, end_idx, instances, scorer, csv_writer):
    iteration_number = 0  # Initialize iteration counter
    for instance in instances[start_idx:end_idx]:
        try:
            case_id = instance["id"]
            source_text = ' '.join(instance["sources"])
            summary_type = "long"
            reference_summary = instance[f"summary/{summary_type}"]

            if reference_summary is None:
                print("none")
                continue

            scores_source = scorer.score(reference_summary, source_text)

            csv_writer.writerow([
                iteration_number,
                case_id,
                scores_source["rouge1"].precision, scores_source["rouge1"].recall, scores_source["rouge1"].fmeasure,
                scores_source["rouge2"].precision, scores_source["rouge2"].recall, scores_source["rouge2"].fmeasure,
                scores_source["rougeL"].precision, scores_source["rougeL"].recall, scores_source["rougeL"].fmeasure
            ])
            print(".")

        except MemoryError:
            print(f"MemoryError: Skipping case_id {case_id} due to insufficient memory.", end="")
            # Write a record indicating the entry was skipped due to a memory error
            csv_writer.writerow([iteration_number, case_id, "skipped", "skipped", "skipped", "skipped", "skipped", "skipped", "skipped", "skipped", "skipped"])

        finally:
            # Clear memory cache after processing each instance regardless of success or error
            gc.collect()
            iteration_number += 1  # Increment iteration number after each instance
            print(iteration_number)

# Load the dataset
multi_lexsum = load_dataset("allenai/multi_lexsum", name="v20230518")
instances_list = list(multi_lexsum["train"])

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Input for range of instances to process
start_list_no = 0
end_list_no = 530

# Adjust end_list_no to not exceed the list length
end_list_no = min(end_list_no, len(instances_list))
print(end_list_no)

# Prepare CSV file
csv_filename = "rouge_scores_detailed.csv"
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "Case ID", "ROUGE-1 Precision", "ROUGE-1 Recall", "ROUGE-1 F1", "ROUGE-2 Precision", "ROUGE-2 Recall", "ROUGE-2 F1", "ROUGE-L Precision", "ROUGE-L Recall", "ROUGE-L F1"])
    
    # Start the total processing time timer
    total_start_time = time.time()
    
    # Process the specified range of instances
    process_instances(start_list_no, end_list_no, instances_list, scorer, writer)
    
    # End the total processing time timer
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time

print("\nResults have been written to", csv_filename)
print("Total Processing Time:", total_processing_time, "seconds")
