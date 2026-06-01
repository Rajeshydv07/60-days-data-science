import json

with open('day17_loan_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

total_size = 0
cells_with_output = 0
for i, cell in enumerate(nb['cells']):
    if cell.get('outputs'):
        cells_with_output += 1
        for out in cell['outputs']:
            if 'data' in out:
                for fmt, content in out['data'].items():
                    if isinstance(content, list):
                        sz = sum(len(s) for s in content)
                    else:
                        sz = len(content)
                    print(f'  Cell {i} [{fmt}]: {sz:,} chars')
                    total_size += sz

print(f'\nTotal output size: {total_size:,} chars')
print(f'Code cells with outputs: {cells_with_output}')
print(f'Total cells: {len(nb["cells"])}')
