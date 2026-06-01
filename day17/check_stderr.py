"""
Check if the notebook has any 'stderr' stream outputs that could cause rendering issues.
GitHub's renderer should handle warnings, but let's check anyway.
Also check for any other structural issues.
"""
import json

path = 'day17_loan_prediction.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"nbformat: {nb['nbformat']}.{nb['nbformat_minor']}")
print()

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        outputs = cell.get('outputs', [])
        if outputs:
            print(f"Cell {i} outputs ({len(outputs)} total):")
            for j, out in enumerate(outputs):
                ot = out.get('output_type')
                if ot == 'stream':
                    name = out.get('name', '?')
                    text = out.get('text', [])
                    text_str = ''.join(text) if isinstance(text, list) else text
                    print(f"  [{j}] stream/{name}: {repr(text_str[:100])}")
                elif ot in ('display_data', 'execute_result'):
                    data_keys = list(out.get('data', {}).keys())
                    print(f"  [{j}] {ot}: keys={data_keys}")
                else:
                    print(f"  [{j}] {ot}: {list(out.keys())}")
        else:
            print(f"Cell {i}: no outputs")
