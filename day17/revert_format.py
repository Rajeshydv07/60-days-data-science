"""
DEFINITIVE FIX: Revert day17 to nbformat 4.2 (same as all other working notebooks).
Remove cell IDs (not needed in 4.2).
This should make GitHub render it the same way as day12, day14, day15, day16.
"""
import json
import os

path = 'day17_loan_prediction.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Before: nbformat {nb['nbformat']}.{nb['nbformat_minor']}")

# Revert to 4.2
nb['nbformat_minor'] = 2

# Remove cell IDs (not valid in nbformat 4.2)
for cell in nb['cells']:
    if 'id' in cell:
        del cell['id']

# Restore metadata to minimal form (same as other days)
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

# Save with same indent as other working notebooks
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

final_size = os.path.getsize(path)
print(f"After: nbformat {nb['nbformat']}.{nb['nbformat_minor']}")
print(f"Final size: {final_size:,} bytes ({final_size//1024} KB)")

# Quick sanity check on PNG sizes
print("\nPNG sizes:")
with open(path, 'r', encoding='utf-8') as f:
    nb2 = json.load(f)
for i, cell in enumerate(nb2['cells']):
    for out in cell.get('outputs', []):
        if 'image/png' in out.get('data', {}):
            sz = len(out['data']['image/png'])
            print(f"  Cell {i}: {sz:,} chars ({sz//1024} KB)")

print("\nDone!")
