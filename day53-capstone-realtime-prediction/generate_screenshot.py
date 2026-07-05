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
    ax.text(10, 85, "Overview", color="white", fontsize=12, ha='center')
    ax.text(10, 78, "Real-Time Prediction", color="#60A5FA", fontsize=12, ha='center')
    ax.text(10, 71, "Churn Analysis", color="white", fontsize=12, ha='center')
    
    # Main Header
    ax.text(25, 92, "Customer Intelligence Platform", color="#1E3A8A", fontsize=24, fontweight='bold')
    ax.text(25, 87, "Live Customer Churn Prediction", color="#374151", fontsize=16, fontweight='bold')
    
    # Form Area
    form_bg = patches.FancyBboxPatch((25, 45), 70, 35, boxstyle="round,pad=1", linewidth=1, edgecolor='#E5E7EB', facecolor='#F9FAFB')
    ax.add_patch(form_bg)
    
    # Inputs
    ax.text(28, 75, "Age", fontsize=10, color="#6B7280")
    ax.add_patch(patches.Rectangle((28, 70), 30, 4, color='white', ec="#D1D5DB"))
    ax.text(29, 71.5, "35", fontsize=10)

    ax.text(60, 75, "Monthly Charges ($)", fontsize=10, color="#6B7280")
    ax.add_patch(patches.Rectangle((60, 70), 30, 4, color='white', ec="#D1D5DB"))
    ax.text(61, 71.5, "85.00", fontsize=10)

    ax.text(28, 62, "Tenure (Months)", fontsize=10, color="#6B7280")
    ax.add_patch(patches.Rectangle((28, 57), 30, 4, color='white', ec="#D1D5DB"))
    ax.text(29, 58.5, "6", fontsize=10)

    ax.text(60, 62, "Contract Type", fontsize=10, color="#6B7280")
    ax.add_patch(patches.Rectangle((60, 57), 30, 4, color='white', ec="#D1D5DB"))
    ax.text(61, 58.5, "Month-to-month", fontsize=10)
    
    # Button
    ax.add_patch(patches.Rectangle((28, 48), 20, 5, color='#2563EB'))
    ax.text(38, 49.5, "Predict Churn Risk", color="white", fontsize=10, ha='center', fontweight='bold')
    
    # Prediction Result Box (High Risk)
    pred_box = patches.FancyBboxPatch((25, 20), 70, 20, boxstyle="round,pad=1", linewidth=1, edgecolor='#FCA5A5', facecolor='#FEE2E2')
    ax.add_patch(pred_box)
    ax.add_patch(patches.Rectangle((25, 19), 1.5, 24, color='#EF4444')) # left border
    
    ax.text(29, 36, "High Churn Risk Detected (90.0%)", color="#991B1B", fontsize=14, fontweight='bold')
    ax.text(29, 31, "This customer has a high likelihood of churning. Immediate action recommended.", color="#7F1D1D", fontsize=10)
    ax.text(29, 25, "• Offer a targeted 15% discount for upgrading to an annual contract.\n• Schedule a proactive outreach call from Customer Success.", color="#7F1D1D", fontsize=10)
    
    # Save the mock screenshot
    plt.tight_layout()
    plt.savefig('ui_screenshot.png', dpi=150, bbox_inches='tight', facecolor='white')

if __name__ == "__main__":
    create_mockup()
    print("Mock screenshot generated as ui_screenshot.png")
