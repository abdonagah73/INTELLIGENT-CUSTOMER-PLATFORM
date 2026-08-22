import os
import zipfile
import subprocess
import pandas as pd
import numpy as np

def run_command(command):
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Command succeeded: {' '.join(command)}")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(command)}")
        print(e.stderr)
        return False

def extract_zip(zip_path, extract_to):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted {zip_path} to {extract_to}")
        os.remove(zip_path)
        return True
    return False

def generate_mock_telco(path):
    print("Generating mock Telco Customer Churn dataset...")
    os.makedirs(path, exist_ok=True)
    np.random.seed(42)
    n = 200 # Small size for mock run
    
    genders = ["Male", "Female"]
    yes_no = ["Yes", "No"]
    contracts = ["Month-to-month", "One year", "Two year"]
    payments = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    internets = ["DSL", "Fiber optic", "No"]
    multiple_lines = ["No phone service", "No", "Yes"]
    internet_add_ons = ["Yes", "No", "No internet service"]
    
    # Generate customer IDs that overlap with bank and support tickets
    customer_ids = [f"{i:04d}-ABCD" for i in range(1, n + 1)]
    
    data = {
        "customerID": customer_ids,
        "gender": np.random.choice(genders, n),
        "SeniorCitizen": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        "Partner": np.random.choice(yes_no, n),
        "Dependents": np.random.choice(yes_no, n),
        "tenure": np.random.randint(0, 73, n),
        "PhoneService": np.random.choice(yes_no, n, p=[0.9, 0.1]),
        "MultipleLines": np.random.choice(multiple_lines, n),
        "InternetService": np.random.choice(internets, n),
        "OnlineSecurity": np.random.choice(internet_add_ons, n),
        "OnlineBackup": np.random.choice(internet_add_ons, n),
        "DeviceProtection": np.random.choice(internet_add_ons, n),
        "TechSupport": np.random.choice(internet_add_ons, n),
        "StreamingTV": np.random.choice(internet_add_ons, n),
        "StreamingMovies": np.random.choice(internet_add_ons, n),
        "Contract": np.random.choice(contracts, n),
        "PaperlessBilling": np.random.choice(yes_no, n),
        "PaymentMethod": np.random.choice(payments, n),
        "MonthlyCharges": np.round(np.random.uniform(18.0, 118.0, n), 2),
        "TotalCharges": [],
        "Churn": np.random.choice(yes_no, n, p=[0.81, 0.19]) # ~19% churn rate
    }
    
    # Let some tenure=0 rows have empty strings in TotalCharges, others have valid strings
    for i in range(n):
        ten = data["tenure"][i]
        mon = data["MonthlyCharges"][i]
        if ten == 0 and np.random.rand() < 0.8:
            data["TotalCharges"].append(" ") # empty string
        else:
            data["TotalCharges"].append(str(np.round(ten * mon * np.random.uniform(0.95, 1.05), 2)))
            
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(path, "WA_Fn-UseC_-Telco-Customer-Churn.csv"), index=False)
    print("Mock Telco dataset saved.")

def generate_mock_bank(path):
    print("Generating mock Bank Customer Churn dataset...")
    os.makedirs(path, exist_ok=True)
    np.random.seed(42)
    n = 200
    
    # Overlap customer IDs with Telco
    customer_ids = [f"{i:04d}-ABCD" for i in range(1, n + 50)] # Some bank customers not in Telco
    
    data = {
        "RowNumber": list(range(1, len(customer_ids) + 1)),
        "CustomerId": customer_ids,
        "Surname": [f"Surname_{i}" for i in range(len(customer_ids))],
        "CreditScore": np.random.randint(350, 850, len(customer_ids)),
        "Geography": np.random.choice(["France", "Spain", "Germany"], len(customer_ids)),
        "Gender": np.random.choice(["Male", "Female"], len(customer_ids)),
        "Age": np.random.randint(18, 90, len(customer_ids)),
        "Tenure": np.random.randint(0, 11, len(customer_ids)),
        "Balance": np.round(np.random.uniform(0.0, 250000.0, len(customer_ids)), 2),
        "NumOfProducts": np.random.randint(1, 5, len(customer_ids)),
        "HasCrCard": np.random.choice([0, 1], len(customer_ids)),
        "IsActiveMember": np.random.choice([0, 1], len(customer_ids)),
        "EstimatedSalary": np.round(np.random.uniform(10000.0, 200000.0, len(customer_ids)), 2),
        "Exited": np.random.choice([0, 1], len(customer_ids), p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(path, "Churn_Modelling.csv"), index=False)
    print("Mock Bank dataset saved.")

def generate_mock_tickets(path):
    print("Generating mock Customer Support Tickets dataset...")
    os.makedirs(path, exist_ok=True)
    np.random.seed(42)
    n = 150
    
    # Random sample of customer IDs (some in Telco, some not)
    customer_ids = [f"{np.random.randint(1, 250):04d}-ABCD" for _ in range(n)]
    
    subjects = [
        "Billing error on my invoice",
        "Internet is extremely slow",
        "Competitor is offering a cheaper plan",
        "I want to cancel my subscription",
        "Network outage in my area",
        "Charged twice for payment",
        "Router not turning on",
        "Requesting a refund for downtime",
        "My contract is ending, any discounts?"
    ]
    
    descriptions = [
        "I was billed $80 instead of $50 as per my contract. Please correct this immediately. This is billing issue.",
        "My internet speed is less than 5 Mbps and I am paying for Fiber 100. This is terrible performance.",
        "Verizon is offering a similar plan for $40/month. Can you match this price or I will switch to competitor.",
        "Please cancel my account at the end of the month. I am moving out of the country.",
        "We have had no internet connection since this morning. Network outage is unacceptable.",
        "My credit card was charged twice for the last bill. Please refund the duplicate charge.",
        "The router you sent me has no power light and won't turn on. I need a replacement device.",
        "My service was down for 3 days last week. I want a credit on my next bill for the disruption.",
        "My 1-year contract is expiring next week. If you can't offer a discount, I am leaving."
    ]
    
    priorities = ["Low", "Medium", "High", "Critical"]
    products = ["Internet", "Phone", "TV", "Multiple Services"]
    statuses = ["Open", "In Progress", "Resolved", "Closed"]
    channels = ["Email", "Web", "Chat", "Phone"]
    
    data = {
        "Ticket ID": [f"TKT-{i:05d}" for i in range(1, n + 1)],
        "Customer Name": [f"User_{i}" for i in range(n)],
        "Customer Email": [f"user_{i}@example.com" for i in range(n)],
        "Customer Age": np.random.randint(18, 75, n),
        "Customer Gender": np.random.choice(["Male", "Female"], n),
        "Product Purchased": np.random.choice(products, n),
        "Date of Purchase": ["2023-01-15"] * n,
        "Ticket Type": ["Complaint"] * n,
        "Ticket Subject": np.random.choice(subjects, n),
        "Ticket Description": np.random.choice(descriptions, n),
        "Ticket Status": np.random.choice(statuses, n),
        "Ticket Priority": np.random.choice(priorities, n),
        "Ticket Channel": np.random.choice(channels, n),
        "Ticket Creation Date": ["2024-02-15"] * n,
        "First Response Time": ["2024-02-15 10:00:00"] * n,
        "Resolution Time": ["2024-02-16 12:00:00"] * n,
        "Customer Satisfaction Rating": np.random.choice([1, 2, 3, 4, 5, None], n),
        "customerID": customer_ids
    }
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(path, "customer_support_tickets.csv"), index=False)
    print("Mock Tickets dataset saved.")

def generate_mock_ecommerce(path):
    print("Generating mock E-Commerce Churn dataset placeholder...")
    os.makedirs(path, exist_ok=True)
    np.random.seed(42)
    n = 50
    customer_ids = [f"{i:04d}-ABCD" for i in range(1, n + 1)]
    
    data = {
        "customerID": customer_ids,
        "logins_per_week": np.random.randint(1, 15, n),
        "cart_abandonment_rate": np.round(np.random.uniform(0.1, 0.9, n), 2),
        "time_since_last_order": np.random.randint(1, 180, n),
        "preferred_device": np.random.choice(["Mobile", "Desktop", "Tablet"], n),
        "preferred_payment_mode": np.random.choice(["Credit Card", "UPI", "Net Banking"], n),
        "days_since_last_order": np.random.randint(1, 180, n)
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(path, "ecommerce_churn_placeholder.csv"), index=False)
    print("Mock E-Commerce placeholder dataset saved.")

def main():
    print("Starting Kaggle dataset ingestion pipeline...")
    
    # Targets
    telco_dir = "./data/raw/telco/"
    bank_dir = "./data/raw/bank/"
    tickets_dir = "./data/raw/tickets/"
    ecommerce_dir = "./data/raw/ecommerce/"
    
    # Try Kaggle CLI download if possible
    kaggle_available = False
    try:
        # Check if kaggle command exists
        subprocess.run(["kaggle", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Check if kaggle credentials are set
        if os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")) or ("KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ):
            kaggle_available = True
    except FileNotFoundError:
        pass
        
    if kaggle_available:
        print("Kaggle CLI and credentials detected. Proceeding with live downloads...")
        
        # 1. Telco
        os.makedirs(telco_dir, exist_ok=True)
        if run_command(["kaggle", "datasets", "download", "-d", "blastchar/telco-customer-churn", "-p", telco_dir]):
            extract_zip(os.path.join(telco_dir, "telco-customer-churn.zip"), telco_dir)
            
        # 2. Bank
        os.makedirs(bank_dir, exist_ok=True)
        if run_command(["kaggle", "datasets", "download", "-d", "gauravtopre/bank-customer-churn-dataset", "-p", bank_dir]):
            extract_zip(os.path.join(bank_dir, "bank-customer-churn-dataset.zip"), bank_dir)
            
        # 3. Tickets
        os.makedirs(tickets_dir, exist_ok=True)
        if run_command(["kaggle", "datasets", "download", "-d", "suraj520/customer-support-ticket-dataset", "-p", tickets_dir]):
            extract_zip(os.path.join(tickets_dir, "customer-support-ticket-dataset.zip"), tickets_dir)
            
        # 4. E-commerce skeleton placeholder
        generate_mock_ecommerce(ecommerce_dir)
    else:
        print("Kaggle credentials not detected or CLI missing. Falling back to local synthetic mock dataset generation...")
        generate_mock_telco(telco_dir)
        generate_mock_bank(bank_dir)
        generate_mock_tickets(tickets_dir)
        generate_mock_ecommerce(ecommerce_dir)
        
    print("Ingestion pipeline completed.")

if __name__ == "__main__":
    main()
