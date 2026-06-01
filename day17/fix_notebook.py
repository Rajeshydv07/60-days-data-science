"""
Fix the day17 notebook for GitHub rendering:
1. Upgrade nbformat_minor from 2 to 5
2. Add cell IDs (required by nbformat 4.5)
3. Fix the decision tree cell - replace huge inline PNG with the saved file reference
4. Ensure all output sizes are reasonable
"""
import json
import uuid
import base64
import os

path = 'day17_loan_prediction.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Before: nbformat={nb['nbformat']}.{nb['nbformat_minor']}")
print(f"Total cells: {len(nb['cells'])}")

# Fix 1: Upgrade nbformat_minor to 5
nb['nbformat_minor'] = 5

# Fix 2: Add cell IDs (required by nbformat 4.5+)
for cell in nb['cells']:
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())[:8]

# Fix 3: Check PNG sizes and replace oversized ones
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        for out in cell.get('outputs', []):
            if 'data' in out and 'image/png' in out['data']:
                png = out['data']['image/png']
                size = len(png)
                print(f"Cell {i}: PNG size = {size:,} chars")
                
                # If decision tree cell (the big one > 50000 chars), replace with saved file
                if size > 50000:
                    print(f"  -> Replacing oversized PNG in cell {i} with saved file")
                    # Read from saved file
                    if os.path.exists('decision_tree_structure.png'):
                        with open('decision_tree_structure.png', 'rb') as img_f:
                            img_bytes = img_f.read()
                        # Compress - resize the saved PNG to small version
                        # For now just use it as-is but encoded properly
                        b64 = base64.b64encode(img_bytes).decode('ascii')
                        # Add newline every 76 chars (standard base64 line wrapping)
                        wrapped = '\n'.join(b64[i:i+76] for i in range(0, len(b64), 76)) + '\n'
                        out['data']['image/png'] = wrapped
                        print(f"  -> New size: {len(wrapped):,} chars")
                    else:
                        print("  -> decision_tree_structure.png not found, keeping as-is")

# Fix 4: Ensure metadata is complete
nb['metadata'] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbformat_minor": "5",
        "pygments_lexer": "ipython3",
        "version": "3.10.0"
    }
}

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

final_size = os.path.getsize(path)
print(f"\nSaved. Final size: {final_size:,} bytes ({final_size/1024:.1f} KB)")
print(f"nbformat: {nb['nbformat']}.{nb['nbformat_minor']}")
