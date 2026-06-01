import json

with open('day17_loan_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("nbformat:", nb.get('nbformat'))
print("nbformat_minor:", nb.get('nbformat_minor'))
print("metadata keys:", list(nb.get('metadata', {}).keys()))
print("kernelspec:", nb['metadata'].get('kernelspec'))
print()
print("Total cells:", len(nb['cells']))
print()

# Check cell structure
for i, cell in enumerate(nb['cells'][:5]):
    ct = cell['cell_type']
    has_id = 'id' in cell
    keys = list(cell.keys())
    print(f"Cell {i}: cell_type={ct}, has_id={has_id}, keys={keys}")

print()
# Check all cells for issues
print("=== Checking all code cells ===")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        outputs = cell.get('outputs', [])
        ec = cell.get('execution_count')
        print(f"Cell {i}: execution_count={ec}, num_outputs={len(outputs)}")
        for j, out in enumerate(outputs):
            ot = out.get('output_type', 'MISSING')
            keys = list(out.keys())
            print(f"  Output {j}: output_type={ot}, keys={keys}")

print()
# Check for any issues with the decision tree cell (large figure)
print("=== Cell 25 (decision tree) output size ===")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and cell.get('outputs'):
        for out in cell['outputs']:
            if 'image/png' in out.get('data', {}):
                png_data = out['data']['image/png']
                print(f"Cell {i}: PNG data type={type(png_data).__name__}, length={len(png_data)}")
