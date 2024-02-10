import json
from collections import Counter

# List of submission files (add the path to your files here)
submission_files = [
    'results/submission.json',
    # ...
]

# Path to the main answer file (set this to your main answer file)
main_answer_file = submission_files[0]

# Load the main answer file
with open(main_answer_file, 'r') as file:
    main_answers = json.load(file)

# Initialize a dictionary to hold all votes
votes = {category: {item["question_id"]: [] for item in questions} for category, questions in main_answers.items()}

# Collect votes from all submission files
for file_path in submission_files:
    with open(file_path, 'r') as file:
        data = json.load(file)
        for category in votes:
            for item in data[category]:
                votes[category][item["question_id"]].append(item["answer"])

# Initialize a dictionary to hold final answers
final_answers = {category: [] for category in votes}

# Perform voting and resolve ties using the main answer file
for category, questions in votes.items():
    for q_id, answer_list in questions.items():
        # Count occurrences of each answer and find the one with the highest count
        answer_counter = Counter(answer_list)
        most_common = answer_counter.most_common(1)

        if len(most_common) == 0 or (len(most_common) == 1 and most_common[0][1] < len(submission_files)/2):
            # No majority or tie: defer to main answer file
            main_answer = next(item for item in main_answers[category] if item["question_id"] == q_id)["answer"]
            final_answer = main_answer
        else:
            # Clear winner
            final_answer = most_common[0][0]

        # Append final answer to the corresponding category
        final_answers[category].append({"question_id": q_id, "answer": final_answer})

# Output final submission
final_submission_file = './results/submission.json'
with open(final_submission_file, 'w') as file:
    json.dump(final_answers, file, indent=4)

print(f"Final submission created: {final_submission_file}")
