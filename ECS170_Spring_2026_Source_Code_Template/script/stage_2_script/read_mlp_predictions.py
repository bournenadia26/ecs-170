import pickle

# Path to the result file
target_file = "../../result/stage_2_result/MLP_prediction_result_0"

with open(target_file, "rb") as f:
    predictions = pickle.load(f)

print("Predictions:")
print(predictions)
print(f"Total predictions: {len(predictions)}")
