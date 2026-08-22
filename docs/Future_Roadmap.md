# Future Product Roadmap

This document outlines the six future product enhancements to expand the capabilities of the customer intelligence platform, detailing the technical approach, estimated effort, and expected business impact.

---

## 1. A/B Model Testing
- **Approach**: Configure a multi-model endpoint routing system (using standard routing proxies or Azure ML Traffic Allocation). Route 10% of API query traffic to the newly retrained model version and 90% to the production baseline. Track model performance metrics and customer renewal rates over a 2-week testing window before promoting the retrained model.
- **Estimated Effort**: 2 weeks (1 engineer)
- **Business Impact**: High. Eliminates staging-to-production deployment risk by verifying model behavior on live, non-simulated traffic.

---

## 2. Customer Segmentation (Clustering)
- **Approach**: Apply unsupervised learning (K-Means clustering or GMM) to the feature store demographic and behavioral tables. Group customers into distinct profiles: (a) High-value loyalty clients, (b) Price-sensitive contract switchers, (c) Feature-frustrated service ticket submitters. Train individual, specialized XGBoost models for each segment.
- **Estimated Effort**: 3 weeks (1 ML engineer)
- **Business Impact**: High. Boosts model precision and allows coordinators to tailor interventions specifically to segment behaviors.

---

## 3. Temporal Churn Forecasting (TFT)
- **Approach**: Replace the binary classification XGBoost model with a sequence-to-sequence model like a Temporal Fusion Transformer (TFT). Utilize historical login dates and charge changes to predict *when* in the next 30 days a customer is most likely to churn.
- **Estimated Effort**: 6 weeks (2 ML engineers)
- **Business Impact**: Very High. Transforms retention campaigns from reactive outreach to proactive, planned engagement.

---

## 4. Graph Neural Networks (GNN)
- **Approach**: Build a customer connection graph where nodes are customerIDs and edges represent connections (shared accounts, family plan dependents, or partner columns). Apply PyTorch Geometric to train a GNN. Churn propagation: if a node churns, increase the risk scores of all connected neighbor nodes.
- **Estimated Effort**: 5 weeks (1 Graph ML engineer)
- **Business Impact**: Medium-High. Captures group-churn dynamics (family plans leaving together).

---

## 5. Causal Inference (EconML)
- **Approach**: Integrate Microsoft's EconML library to estimate the **Heterogeneous Treatment Effect**. Move from predicting *who* will churn to predicting *which* specific discount or call will have the highest positive impact on keeping the customer.
- **Estimated Effort**: 4 weeks (1 ML engineer)
- **Business Impact**: Very High. Prevents wasting discounts on customers who would stay anyway, boosting campaign efficiency.

---

## 6. Cross-Region Azure Deployment
- **Approach**: Replicate feature store blobs across Azure West Europe and US East. Use local endpoints for low-latency scoring and synchronize database logs to a centralized server for training cycles.
- **Estimated Effort**: 3 weeks (1 Cloud engineer)
- **Business Impact**: Medium. Prepares the infrastructure to scale globally.
