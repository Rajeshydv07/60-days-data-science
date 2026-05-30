"""
Strip all outputs from day17 notebook so GitHub can reliably render it.
GitHub can ALWAYS render notebooks with no outputs - the error only 
happens when output data is present and the renderer times out or fails.

A clean notebook with code + markdown + no outputs renders perfectly on GitHub.
"""
import json

path = 'c:/60-days-data-science/day17/day17_loan_prediction.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Before: {len(nb['cells'])} cells")
total_outputs_removed = 0

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        n = len(cell.get('outputs', []))
        cell['outputs'] = []
        cell['execution_count'] = None
        total_outputs_removed += n

print(f"Removed {total_outputs_removed} total outputs")

# Ensure correct format
nb['nbformat'] = 4
nb['nbformat_minor'] = 2
nb['metadata'] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.10.0"
    }
}

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

import os
size_kb = os.path.getsize(path) / 1024
print(f"Saved. New file size: {size_kb:.1f} KB")
print("Done - notebook now has clean code + markdown, no outputs.")
