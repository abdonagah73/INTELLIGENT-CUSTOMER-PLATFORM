# Data Dictionary

This document details every column in the central versioned Feature Store, listing the source group, data type, null rate, and business meaning.

---

## 1. Demographic Feature Group
Stored in: `/data/feature_store/demographic/snapshot_[date].parquet`

| Feature Name | Source | Data Type | Null Rate | Business Meaning / Values |
| :--- | :--- | :--- | :--- | :--- |
| `customerID` | Telco | String | 0.0% | Unique alphanumeric customer identifier (primary join key) |
| `gender` | Telco | String | 0.0% | Customer gender (`Male`, `Female`) |
| `SeniorCitizen` | Telco | Integer | 0.0% | Indicator if customer is a senior citizen (`1` = Yes, `0` = No) |
| `Partner` | Telco | Integer | 0.0% | Indicator if customer has a partner (`1` = Yes, `0` = No) |
| `Dependents` | Telco | Integer | 0.0% | Indicator if customer has dependents (`1` = Yes, `0` = No) |
| `tenure` | Telco | Integer | 0.0% | Months customer has stayed with company (0 to 72) |
| `tenure_band` | Engineered | String | 0.0% | Segmented tenure ranges (`<12 months`, `12-24 months`, `24-48 months`, `>48 months`) |
| `CreditScore` | Bank | Integer | 0.0% | Credit score of the customer (300 to 850, filled with median if missing) |
| `Balance` | Bank | Double | 0.0% | Customer account/billing balance (filled with median if missing) |
| `NumOfProducts` | Bank | Integer | 0.0% | Number of financial/telecom products subscribed to (filled with median if missing) |
| `EstimatedSalary` | Bank | Double | 0.0% | Estimated salary of the customer (filled with median if missing) |
| `Churn` | Telco | Integer | 0.0% | Target variable: Churn indicator (`1` = Yes, `0` = No) |

---

## 2. Behavioral Feature Group
Stored in: `/data/feature_store/behavioral/snapshot_[date].parquet`

| Feature Name | Source | Data Type | Null Rate | Business Meaning / Values |
| :--- | :--- | :--- | :--- | :--- |
| `customerID` | Telco | String | 0.0% | Unique alphanumeric customer identifier |
| `service_bundle_size` | Engineered | Integer | 0.0% | Number of active add-on services (Phone, Internet, Tech Support, etc.) |
| `TechSupport` | Telco | Integer | 0.0% | Customer has technical support add-on (`1` = Yes, `0` = No) |
| `OnlineSecurity` | Telco | Integer | 0.0% | Customer has online security add-on (`1` = Yes, `0` = No) |
| `logins_7d_rolling_avg` | Synthetic Stream | Double | 0.0% | Average number of login events per day over the last 7 days |
| `logins_30d_rolling_avg` | Synthetic Stream | Double | 0.0% | Average number of login events per day over the last 30 days |
| `login_trend` | Synthetic Stream | Double | 0.0% | Change in login behavior (`logins_7d_rolling_avg` - `logins_30d_rolling_avg`) |
| `support_contacts_30d` | Synthetic Stream | Integer | 0.0% | Number of support tickets submitted by customer in the last 30 days |
| `days_since_last_login` | Synthetic Stream | Integer | 0.0% | Number of days since the customer last logged into their account portal |
| `usage_delta_mom` | Synthetic Stream | Double | 0.0% | Month-over-month usage volume variance percentage |
| `sentiment_score` | LLM Extracted | Double | 0.0% | Aggregated sentiment score of customer tickets (-1.0 = negative, 1.0 = positive) |
| `urgency_level` | LLM Extracted | String | 0.0% | Highest urgency level of customer tickets (`low`, `medium`, `high`, `critical`) |
| `escalation_flag` | LLM Extracted | Integer | 0.0% | Set to `1` if any customer ticket threatens cancellation, competitor switch, or legal action |
| `public_complaint_flag` | Twitter (Skeleton) | Integer | 0.0% | Customer posted a public social media complaint (Placeholder, default `0`) |

---

## 3. Billing Feature Group
Stored in: `/data/feature_store/billing/snapshot_[date].parquet`

| Feature Name | Source | Data Type | Null Rate | Business Meaning / Values |
| :--- | :--- | :--- | :--- | :--- |
| `customerID` | Telco | String | 0.0% | Unique alphanumeric customer identifier |
| `MonthlyCharges` | Telco | Double | 0.0% | Current monthly billing charges in USD |
| `TotalCharges` | Telco | Double | 0.0% | Cumulative billing charges in USD (empty strings resolved to tenure * MonthlyCharges) |
| `charges_per_tenure_month`| Engineered | Double | 0.0% | Average billing cost per active tenure month (`TotalCharges` / `tenure`) |
| `billing_delta_mom` | Engineered | Double | 0.0% | Simulated change in month-over-month billing amount |
| `contract_type_encoded` | Telco | Integer | 0.0% | Ordinally encoded contract term length (`0` = Month-to-month, `1` = One year, `2` = Two year) |
| `PaperlessBilling` | Telco | Integer | 0.0% | Customer uses paperless billing options (`1` = Yes, `0` = No) |
| `is_high_risk_contract` | Engineered | Integer | 0.0% | Flags customers with month-to-month contracts and < 12 months tenure (`1` = Yes, `0` = No) |
| `PaymentMethod` | Telco | String | 0.0% | Raw payment method string (scaled during model preprocessing) |
| `InternetService` | Telco | String | 0.0% | Raw internet type string (scaled during model preprocessing) |

---

## 4. LLM Extracted Feature Group
Stored in: `/data/feature_store/llm_features/snapshot_[date].parquet`

| Feature Name | Source | Data Type | Null Rate | Business Meaning / Values |
| :--- | :--- | :--- | :--- | :--- |
| `customerID` | Telco | String | 0.0% | Unique alphanumeric customer identifier |
| `ticket_sentiment_score` | LLM Extracted | Double | 0.0% | Average sentiment score across customer support history |
| `ticket_urgency_level` | LLM Extracted | String | 0.0% | Max urgency rating across customer support history |
| `ticket_complaint_topics_billing` | LLM Extracted | Integer | 0.0% | Customer has submitted a billing-related complaint (`1` = Yes, `0` = No) |
| `ticket_complaint_topics_competitor` | LLM Extracted | Integer | 0.0% | Customer has submitted a competitor-related complaint (`1` = Yes, `0` = No) |
| `ticket_complaint_topics_performance`| LLM Extracted | Integer | 0.0% | Customer has submitted a performance-related complaint (`1` = Yes, `0` = No) |
| `ticket_escalation_flag` | LLM Extracted | Integer | 0.0% | Set to `1` if any customer ticket threatens cancellation, competitor switch, or legal action |
| `review_sentiment_monthly` | Web Scraping | Double | 0.0% | Public Trustpilot review sentiment placeholder |

---

## 5. Market Signals Feature Group
Stored in: `/data/feature_store/market_signals/snapshot_[date].parquet`

| Feature Name | Source | Data Type | Null Rate | Business Meaning / Values |
| :--- | :--- | :--- | :--- | :--- |
| `customerID` | Telco | String | 0.0% | Unique alphanumeric customer identifier |
| `competitor_news_volume_7d` | NewsAPI | Integer | 0.0% | Number of mentions of competitor brands in news media over the last 7 days |
| `public_review_sentiment_monthly` | Web Scraping | Double | 0.0% | Aggregated monthly Trustpilot company review sentiment score |
| `one_star_review_rate_monthly` | Web Scraping | Double | 0.0% | Ratio of 1-star reviews received by company in the last 30 days |
| `price_competitiveness_ratio` | Web Scraping | Double | 0.0% | Ratio of competitor cheapest price / customer monthly charges |
| `competitor_promotion_flag` | Web Scraping | Boolean | 0.0% | Indicator if competitors are running aggressive discount campaigns |
