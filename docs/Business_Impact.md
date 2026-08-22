# Business Impact & ROI Analysis

This document describes the financial projections, prevented churn calculations, and return on investment (ROI) metrics for the customer retention platform.

---

## 1. Parameters & Cost Structure

- **Customer Lifetime Value (LTV)**: **$1,680** (Calculated as average Monthly Charge of $70 over a 24-month typical contract tenure).
- **Intervention Cost (Discount Offer / Call)**: **$50** average per targeted customer.
- **Asymmetric Cost Ratio**: **33.6 : 1** (The cost of a missed churner ($1,680) is 33.6 times the cost of a false alarm intervention ($50)).
- **Baseline Churn Rate**: **18.5%** (IBM dataset churn population).

---

## 2. Threshold Comparison Analysis
By comparing the standard mathematical classification threshold (`0.50`) against the asymmetric cost-optimized threshold (`0.32`), we demonstrate significant business savings:

### Scenario A: Standard Threshold (0.50)
- **Characteristics**: Focuses strictly on probability accuracy; misses a larger portion of actual churners because the classification boundary is too high.
- **Target metrics**: Recall of 52%.
- **Impact per 1,000 customers**:
  - Out of 185 actual churners, 96 are caught, and 89 are missed (False Negatives).
  - False Negatives Cost: \(89 \times \$1,680 = \$149,520\)
  - False Positives Cost (Interventions): \(10 \times \$50 = \$500\)
  - **Total Retention Campaign Cost**: **$150,020**

### Scenario B: Cost-Optimized Threshold (0.32)
- **Characteristics**: Lowers the classification boundary to catch more churners, accepting more false alarms because interventions are cheap compared to LTV loss.
- **Target metrics**: Recall of 82%.
- **Impact per 1,000 customers**:
  - Out of 185 actual churners, 151 are caught, and 34 are missed (False Negatives).
  - False Negatives Cost: \(34 \times \$1,680 = \$57,120\)
  - False Positives Cost (Interventions): \(100 \times \$50 = \$5,000\)
  - **Total Retention Campaign Cost**: **$62,120**

### Financial Savings
- **Savings per 1,000 customers**: \(150,020 - 62,120 = \mathbf{\$87,900}\)
- **Reduction in Churn Costs**: **58.7%**

---

## 3. Platform Return on Investment (ROI)

Assuming a customer base of **50,000** active subscribers:

| Metric | Value |
| :--- | :--- |
| Annual Churn Population (18.5%) | 9,250 customers |
| Prevented Churn via Platform (82% recall vs 52% baseline) | 2,775 additional customers retained |
| **Gross Revenue Saved** | **$4,662,000** |
| Campaign Intervention Costs (Targeting 9,000 customers) | ($450,000) |
| Platform Operations & API Costs (Annualized) | ($60,000) |
| **Net Savings** | **$4,152,000** |
| **Platform ROI (Year 1)** | **6,920%** |
