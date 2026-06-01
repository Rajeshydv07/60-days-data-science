import json

# Deep structural comparison - check every field in day17 cells vs day16
with open('c:/60-days-data-science/day17/day17_loan_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb17 = json.load(f)

with open('c:/60-days-data-science/day16/day16_movie_recommendation.ipynb', 'r', encoding='utf-8') as f:
    nb16 = json.load(f)

print("=== Day17 metadata ===")
print(json.dumps(nb17['metadata'], indent=2))

print("\n=== Day16 metadata ===")
print(json.dumps(nb16['metadata'], indent=2))

print("\n=== Day17 first code cell structure ===")
for cell in nb17['cells']:
    if cell['cell_type'] == 'code':
        print("keys:", list(cell.keys()))
        print("metadata:", cell.get('metadata'))
        if cell.get('outputs'):
            out = cell['outputs'][0]
            print("output keys:", list(out.keys()))
        break

print("\n=== Day16 first code cell structure ===")
for cell in nb16['cells']:
    if cell['cell_type'] == 'code':
        print("keys:", list(cell.keys()))
        print("metadata:", cell.get('metadata'))
        if cell.get('outputs'):
            out = cell['outputs'][0]
            print("output keys:", list(out.keys()))
        break

# Check if any cell in day17 has 'id' field still
print("\n=== Checking for 'id' fields in day17 cells ===")
for i, cell in enumerate(nb17['cells']):
    if 'id' in cell:
        print(f"Cell {i} still has id: {cell['id']}")
print("Done checking ids")

# Print structure of large tree plot cell output
print("\n=== Cell 25 (decision tree) output structure ===")
cell = nb17['cells'][25]
print("Cell type:", cell['cell_type'])
if cell.get('outputs'):
    out = cell['outputs'][0]
    print("Output keys:", list(out.keys()))
    print("Output type:", out.get('output_type'))
    if 'data' in out:
        for mime, val in out['data'].items():
            print(f"  {mime}: {len(str(val))} bytes (type={type(val).__name__})")
            if isinstance(val, str):
                print(f"  First 100 chars: {val[:100]}")
