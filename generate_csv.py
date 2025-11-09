import pandas as pd
import requests
import ast  # for safely evaluating string to Python object

# -----------------------------
# Your test queries
# -----------------------------
test_queries = [
    "I am hiring for Java developers who can collaborate effectively.",
    "Looking to hire mid-level professionals proficient in Python, SQL and JavaScript.",
    "JD text: recommend assessments for analysts using cognitive and personality tests."
]

# -----------------------------
# API endpoint
# -----------------------------
API_URL = "http://localhost:8000/recommend"  # or your deployed endpoint

# Prepare lists to build CSV
csv_data = {"Query": [], "Assessment_url": []}

for query in test_queries:
    # Call your FastAPI endpoint
    response = requests.post(API_URL, json={"query": query})
    recommendations_str = response.json().get("recommendations", "")  # this is a string

    try:
        # Convert string output to Python list of dicts
        recommendations = ast.literal_eval(recommendations_str)
    except Exception as e:
        print(f"Failed to parse API response for query: {query}")
        print("Response:", recommendations_str)
        continue

    # Now each rec is a dict
    for rec in recommendations:
        csv_data["Query"].append(query)
        csv_data["Assessment_url"].append(rec.get("URL", ""))

# Create DataFrame
df = pd.DataFrame(csv_data)

# Save CSV in required format
df.to_csv("shl_assessment_recommendations.csv", index=False)
print("CSV generated successfully!")
