"""
Rebuild day17 notebook from scratch matching the exact JSON structure of 
the working day16 notebook. The issue is likely the `source` field in cells 
being stored as a single string rather than a list of strings (as day16 does).
"""
import json
import base64

# Load the existing executed notebook to get the output data
with open('c:/60-days-data-science/day17/day17_loan_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb17_executed = json.load(f)

# Load day16 to copy its exact structure
with open('c:/60-days-data-science/day16/day16_movie_recommendation.ipynb', 'r', encoding='utf-8') as f:
    nb16 = json.load(f)

# Check how day16 stores cell source
print("=== Day16 source type in cells ===")
for i, cell in enumerate(nb16['cells'][:3]):
    src = cell.get('source', '')
    print(f"Cell {i}: type={type(src).__name__}, cell_type={cell['cell_type']}")
    if isinstance(src, list):
        print(f"  List with {len(src)} items, first item: {repr(src[0][:50])}")
    else:
        print(f"  String: {repr(src[:50])}")

print()
print("=== Day17 source type in cells ===")
for i, cell in enumerate(nb17_executed['cells'][:3]):
    src = cell.get('source', '')
    print(f"Cell {i}: type={type(src).__name__}, cell_type={cell['cell_type']}")
    if isinstance(src, list):
        print(f"  List with {len(src)} items")
    else:
        print(f"  String length: {len(src)}")

print()
print("=== Day16 output data structure for a plot cell ===")
for i, cell in enumerate(nb16['cells']):
    if cell['cell_type'] == 'code' and cell.get('outputs'):
        for out in cell['outputs']:
            if 'data' in out and 'image/png' in out['data']:
                png = out['data']['image/png']
                txt = out['data'].get('text/plain', '')
                print(f"Cell {i}:")
                print(f"  image/png type: {type(png).__name__}, is_list: {isinstance(png, list)}")
                print(f"  text/plain type: {type(txt).__name__}, is_list: {isinstance(txt, list)}")
                if isinstance(png, list):
                    print(f"  png list length: {len(png)}")
                if isinstance(txt, list):
                    print(f"  txt list: {txt}")
                else:
                    print(f"  txt value: {repr(txt)}")
                break
    else:
        continue
    break

print()
print("=== Day17 output data structure for a plot cell ===")
for i, cell in enumerate(nb17_executed['cells']):
    if cell['cell_type'] == 'code' and cell.get('outputs'):
        for out in cell['outputs']:
            if 'data' in out and 'image/png' in out['data']:
                png = out['data']['image/png']
                txt = out['data'].get('text/plain', '')
                print(f"Cell {i}:")
                print(f"  image/png type: {type(png).__name__}, is_list: {isinstance(png, list)}")
                print(f"  text/plain type: {type(txt).__name__}, is_list: {isinstance(txt, list)}")
                if isinstance(png, list):
                    print(f"  png list length: {len(png)}")
                if isinstance(txt, list):
                    print(f"  txt list: {txt}")
                else:
                    print(f"  txt value: {repr(txt)}")
                break
    else:
        continue
    break
