import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_mockup():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Sidebar
    sidebar = patches.Rectangle((0, 0), 20, 100, linewidth=1, edgecolor='none', facecolor='#1F2937')
    ax.add_patch(sidebar)
    ax.text(10, 95, "Analytics Pro", color="white", fontsize=16, fontweight='bold', ha='center')
    ax.text(10, 85, "Overview", color="#60A5FA", fontsize=12, ha='center')
    ax.text(10, 78, "Churn Analysis", color="white", fontsize=12, ha='center')
    ax.text(10, 71, "Customers", color="white", fontsize=12, ha='center')
    
    # Main Header
    ax.text(25, 92, "Customer Intelligence Platform", color="#1E3A8A", fontsize=24, fontweight='bold')
    
    # KPI Cards
    for i, title, val in [(0, "Total Users", "1,000"), (1, "Churn Rate", "25.0%"), (2, "Avg LTV", "$2,500"), (3, "Tenure", "24 mos")]:
        rect = patches.FancyBboxPatch((25 + i*18, 75), 14, 10, boxstyle="round,pad=0.5", linewidth=1, edgecolor='#E5E7EB', facecolor='#F3F4F6')
        ax.add_patch(rect)
        ax.text(25 + i*18 + 7, 82, title, color="#4B5563", fontsize=10, ha='center')
        ax.text(25 + i*18 + 7, 77, val, color="#2563EB", fontsize=16, fontweight='bold', ha='center')
        
    # Charts Area
    chart1 = patches.FancyBboxPatch((25, 30), 32, 38, boxstyle="round,pad=0.5", linewidth=1, edgecolor='#E5E7EB', facecolor='white')
    ax.add_patch(chart1)
    ax.text(41, 65, "Monthly Revenue Dist", fontsize=12, color="#374151")
    ax.bar([30, 33, 36, 39, 42, 45, 48, 51, 54], [5, 10, 15, 25, 20, 10, 8, 4, 2], width=2, color="#10B981")
    
    chart2 = patches.FancyBboxPatch((63, 30), 32, 38, boxstyle="round,pad=0.5", linewidth=1, edgecolor='#E5E7EB', facecolor='white')
    ax.add_patch(chart2)
    ax.text(79, 65, "Customer Segments", fontsize=12, color="#374151")
    pie = patches.Circle((79, 48), radius=10, color="#60A5FA")
    ax.add_patch(pie)
    
    # Save the mock screenshot
    plt.tight_layout()
    plt.savefig('ui_screenshot.png', dpi=150, bbox_inches='tight', facecolor='white')

if __name__ == "__main__":
    create_mockup()
    print("Mock screenshot generated as ui_screenshot.png")
