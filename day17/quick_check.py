import json

path = 'day17_loan_prediction.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"nbformat: {nb['nbformat']}.{nb['nbformat_minor']}")
print(f"Total cells: {len(nb['cells'])}")
print()

for i, cell in enumerate(nb['cells'][:5]):
    ct = cell['cell_type']
    ec = cell.get('execution_count')
    src = ''.join(cell.get('source', []))[:80].replace('\n', ' ')
    num_out = len(cell.get('outputs', []))
    print(f"Cell {i} ({ct}): ec={ec}, outputs={num_out}")
    print(f"  src: {repr(src)}")
