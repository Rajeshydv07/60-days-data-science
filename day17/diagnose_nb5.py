"""
Find the real issue - check cells with image outputs specifically
"""
import json

with open('c:/60-days-data-science/day17/day17_loan_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb17 = json.load(f)

with open('c:/60-days-data-science/day16/day16_movie_recommendation.ipynb', 'r', encoding='utf-8') as f:
    nb16 = json.load(f)

# Find first image output in day16
print("=== Day16 first image cell ===")
for i, cell in enumerate(nb16['cells']):
    if cell.get('outputs'):
        for out in cell['outputs']:
            if out.get('output_type') == 'display_data' and 'image/png' in out.get('data', {}):
                print("Raw JSON of this output (first 500 chars):")
                print(json.dumps(out, indent=2)[:500])
                break
        else:
            continue
        break

print()
print("=== Day17 first image cell ===")
for i, cell in enumerate(nb17['cells']):
    if cell.get('outputs'):
        for out in cell['outputs']:
            if out.get('output_type') == 'display_data' and 'image/png' in out.get('data', {}):
                print("Raw JSON of this output (first 500 chars):")
                print(json.dumps(out, indent=2)[:500])
                break
        else:
            continue
        break

print()
# Check the metadata field in image outputs
print("=== Checking 'metadata' in image outputs ===")
print("Day16 image output metadata:")
for cell in nb16['cells']:
    for out in cell.get('outputs', []):
        if 'image/png' in out.get('data', {}):
            print("  metadata:", json.dumps(out.get('metadata', 'MISSING')))
            break
    else:
        continue
    break

print("Day17 image output metadata:")
for cell in nb17['cells']:
    for out in cell.get('outputs', []):
        if 'image/png' in out.get('data', {}):
            print("  metadata:", json.dumps(out.get('metadata', 'MISSING')))
            break
    else:
        continue
    break
