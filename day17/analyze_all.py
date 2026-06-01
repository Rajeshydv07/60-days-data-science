"""
Check all other day notebooks to find which ones render successfully on GitHub
and what their max PNG sizes are.
"""
import json
import os

base = 'c:/60-days-data-science'

for day_num in range(1, 20):
    day_dir = os.path.join(base, f'day{day_num}')
    if not os.path.exists(day_dir):
        continue
    
    # Find notebook files
    for fname in os.listdir(day_dir):
        if fname.endswith('.ipynb'):
            nb_path = os.path.join(day_dir, fname)
            try:
                with open(nb_path, 'r', encoding='utf-8') as f:
                    nb = json.load(f)
                
                total_size = os.path.getsize(nb_path)
                nbformat = f"{nb.get('nbformat', '?')}.{nb.get('nbformat_minor', '?')}"
                
                max_png = 0
                total_png = 0
                png_count = 0
                for cell in nb.get('cells', []):
                    for out in cell.get('outputs', []):
                        if 'image/png' in out.get('data', {}):
                            sz = len(out['data']['image/png'])
                            max_png = max(max_png, sz)
                            total_png += sz
                            png_count += 1
                
                print(f"day{day_num}/{fname}:")
                print(f"  file={total_size//1024}KB, nbformat={nbformat}, "
                      f"PNGs={png_count}, maxPNG={max_png//1024}KB, totalPNG={total_png//1024}KB")
            except Exception as e:
                print(f"day{day_num}/{fname}: ERROR - {e}")
