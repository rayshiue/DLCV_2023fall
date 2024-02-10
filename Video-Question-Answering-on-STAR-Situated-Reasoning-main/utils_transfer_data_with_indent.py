# transfer json without indent to json with indent

import json

def read_data():
    with open('Flipped_VQA/results/submission.json', 'r') as f:
        data = json.load(f)
    return data
  
def write_data(data):
    with open('submission_example_indent.json', 'w') as f:
        json.dump(data, f, indent=4)
        
if __name__ == '__main__':
    data = read_data()
    write_data(data)
