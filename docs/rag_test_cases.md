# RAG Test Cases

This document lists 10 sample customer cases representing different scenarios, profiles, and support history, detailing the expected RAG briefing strategies from the FAISS database.

---

### Case 1: Month-to-Month Contract, Short Tenure, No Tech Support
- **Customer ID**: `0001-ABCD`
- **Key Characteristics**: tenure = 3 months, Contract = "Month-to-month", TechSupport = "No"
- **Query Query Terms**: `customer month-to-month contract tenure 3 months no tech support`
- **Expected Playbook Section**: Contract Expiry & Month-to-Month Churners
- **Expected Recommendation Action**: Transition them to a yearly contract by offering a free trial of premium add-on services (Online Security and Online Backup) for 3 months.

### Case 2: Long Tenure, Yearly Contract, Low Portal Logins
- **Customer ID**: `0002-ABCD`
- **Key Characteristics**: tenure = 52 months, Contract = "One year", logins_30d_rolling_avg = 0.10
- **Query Query Terms**: `customer one-year contract tenure 52 months`
- **Expected Playbook Section**: Long-Tenure Loyalty Churners
- **Expected Recommendation Action**: Proactive loyalty outreach via account manager; offer free service bundle upgrades (streaming TV/movies) at no cost.

### Case 3: Fiber Optic Internet, High Billing Charges, Tech Complaint
- **Customer ID**: `0003-ABCD`
- **Key Characteristics**: InternetService = "Fiber optic", MonthlyCharges = $110.00, TechSupport = "No"
- **Query Query Terms**: `customer fiber optic internet monthly charges $110.00 no tech support`
- **Expected Playbook Section**: Tech-Issue & Fiber Optic Service Churners
- **Expected Recommendation Action**: Immediate technical dispatch, free Tech Support add-on for 6 months, router check, and bill credit.

### Case 4: Billing Dispute, Double Charge Complaint
- **Customer ID**: `0004-ABCD`
- **Key Characteristics**: MonthlyCharges = $75.00, ticket_complaint_topics_billing = 1, sentiment_score = -0.7
- **Query Query Terms**: `customer monthly charges $75.00 negative ticket sentiment -0.70`
- **Expected Playbook Section**: Billing Disputes Churners
- **Expected Recommendation Action**: Correct billing ledger immediately, issue credit refund, send confirmation within 1 hour, apply $15 credit.

### Case 5: Short Tenure, Month-to-Month, High Competitor Risk
- **Customer ID**: `0005-ABCD`
- **Key Characteristics**: tenure = 5 months, Contract = "Month-to-month", price_competitiveness_ratio = 0.65
- **Query Query Terms**: `customer month-to-month contract tenure 5 months`
- **Expected Playbook Section**: Price-Sensitive Churners / Contract Expiry
- **Expected Recommendation Action**: Offer Loyalty discount or price match; transition to a yearly contract with a price guarantee.

### Case 6: Critical Support Ticket Escalation
- **Customer ID**: `0006-ABCD`
- **Key Characteristics**: escalation_flag = 1, sentiment_score = -0.90, Contract = "Month-to-month"
- **Query Query Terms**: `customer month-to-month contract escalated support ticket negative ticket sentiment -0.90`
- **Expected Playbook Section**: Billing Disputes / Tech-Issue / Contract Expiry
- **Expected Recommendation Action**: Immediate callback, account waiver credit, or support escalation resolver depending on main ticket driver.

### Case 7: High Competitor News Exposure
- **Customer ID**: `0007-ABCD`
- **Key Characteristics**: competitor_news_volume_7d = 25, price_competitiveness_ratio = 0.55
- **Query Query Terms**: `customer monthly charges competitor deals`
- **Expected Playbook Section**: Price-Sensitive Churners
- **Expected Recommendation Action**: Proactive competitive price matching or contract term upgrade offer.

### Case 8: Fiber Optic Internet Customer with Multiple Outages
- **Customer ID**: `0008-ABCD`
- **Key Characteristics**: InternetService = "Fiber optic", support_contacts_30d = 4, sentiment_score = -0.80
- **Query Query Terms**: `customer fiber optic internet negative ticket sentiment -0.80`
- **Expected Playbook Section**: Tech-Issue & Fiber Optic Service Churners
- **Expected Recommendation Action**: Upgrade line filters, waive next monthly statement, next-day replacement router shipping.

### Case 9: Long-Tenure Customer with Pending Billing Ticket
- **Customer ID**: `0009-ABCD`
- **Key Characteristics**: tenure = 60 months, ticket_complaint_topics_billing = 1, MonthlyCharges = $95.00
- **Query Query Terms**: `customer tenure 60 months monthly charges $95.00`
- **Expected Playbook Section**: Long-Tenure Loyalty Churners / Billing Disputes
- **Expected Recommendation Action**: Resolve billing dispute instantly and upgrade service bundle at no extra cost.

### Case 10: New Customer on Two-Year Contract
- **Customer ID**: `0010-ABCD`
- **Key Characteristics**: tenure = 2 months, Contract = "Two year", Churn = 0
- **Query Query Terms**: `customer two-year contract tenure 2 months`
- **Expected Playbook Section**: General retention / stable customer
- **Expected Recommendation Action**: Stable customer classification; no proactive contract modification required.
