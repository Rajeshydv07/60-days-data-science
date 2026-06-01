import json

# Check what outputs look like in day17
with open('c:/60-days-data-science/day17/day17_loan_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb17 = json.load(f)

print('nbformat:', nb17['nbformat'], nb17['nbformat_minor'])
total_output_bytes = 0
for i, cell in enumerate(nb17['cells']):
    if cell['cell_type'] == 'code':
        for j, out in enumerate(cell.get('outputs', [])):
            otype = out.get('output_type', '')
            if 'data' in out:
                for mime, val in out['data'].items():
                    size = len(str(val))
                    total_output_bytes += size
                    if size > 5000:
                        print(f'Cell {i} output {j}: {mime} = {size} bytes')

cells_count = len(nb17['cells'])
print(f'Total output data bytes: {total_output_bytes}')
print(f'Cells count: {cells_count}')

# Check day16 for comparison
with open('c:/60-days-data-science/day16/day16_movie_recommendation.ipynb', 'r', encoding='utf-8') as f:
    nb16 = json.load(f)
print()
print('Day16 nbformat:', nb16['nbformat'], nb16['nbformat_minor'])
total16 = 0
for i, cell in enumerate(nb16['cells']):
    if cell['cell_type'] == 'code':
        for j, out in enumerate(cell.get('outputs', [])):
            if 'data' in out:
                for mime, val in out['data'].items():
                    total16 += len(str(val))
print(f'Day16 total output data bytes: {total16}')
