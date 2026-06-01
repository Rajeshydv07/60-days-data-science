import json

with open('c:/60-days-data-science/day17/day17_loan_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb17 = json.load(f)

with open('c:/60-days-data-science/day16/day16_movie_recommendation.ipynb', 'r', encoding='utf-8') as f:
    nb16 = json.load(f)

print("=== Day17 all output types ===")
for i, cell in enumerate(nb17['cells']):
    if cell['cell_type'] == 'code' and cell.get('outputs'):
        for j, out in enumerate(cell['outputs']):
            otype = out.get('output_type')
            data_keys = list(out.get('data', {}).keys())
            metadata_keys = list(out.get('metadata', {}).keys())
            print(f"  Cell {i} out {j}: output_type={otype}, data={data_keys}, metadata_keys={metadata_keys}")

print()
print("=== Day16 sample output types (first 10 code cells) ===")
count = 0
for i, cell in enumerate(nb16['cells']):
    if cell['cell_type'] == 'code' and cell.get('outputs') and count < 10:
        for j, out in enumerate(cell['outputs']):
            otype = out.get('output_type')
            data_keys = list(out.get('data', {}).keys())
            metadata_keys = list(out.get('metadata', {}).keys())
            if 'image/png' in data_keys:
                print(f"  Cell {i} out {j}: output_type={otype}, data={data_keys}, metadata_keys={metadata_keys}")
                count += 1
