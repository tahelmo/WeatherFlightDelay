import requests
import json
import time
import os
import pandas as pd
from tqdm import tqdm

BASE_URL = "https://www.otstcfq.org/wp-json/members/"
LIST_ENDPOINT = "get_fuse_members/"
DETAIL_ENDPOINT = "get_fuse_member/"
JSON_FILE = "otstcfq_members.json"
CSV_FILE = "otstcfq_members.csv"

def fetch_members_list():
    """Fetch the list of members."""
    try:
        response = requests.get(BASE_URL + LIST_ENDPOINT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching member list: {e}")
        return []

def fetch_member_details(member_id):
    """Fetch detailed info for a single member by ID."""
    try:
        url = BASE_URL + DETAIL_ENDPOINT + str(member_id)
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching details for ID {member_id}: {e}")
        return None

def load_existing_data():
    """Load existing data if file exists (resume support)."""
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_data(data):
    """Save data to JSON file."""
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def flatten_member(details):
    """Flatten member details into a dictionary for CSV export."""
    return {
        "Permit": details.get("member_permit", ""),
        "Full Name": details.get("full_name", ""),
        "First Name": details.get("member_first_name", ""),
        "Last Name": details.get("member_last_name", ""),
        "Gender": details.get("member_gender", ""),
        "Date Limitation": details.get("member_date_limitation", ""),
        "Restriction": details.get("member_restriction", ""),
        "Limitation": details.get("member_limitation", ""),
        "Authorization Type": details.get("member_authorization_type", ""),
        "Professional Title": details.get("member_professional_title", ""),
        "Professional Categories": details.get("member_professional_categories", ""),
        "Languages": details.get("member_languages", ""),
        "Company Name": details.get("business_card_company_name", ""),
        "Work Title": details.get("business_card_work_title", ""),
        "Street": details.get("business_card_address_street", ""),
        "City": details.get("business_card_address_city", ""),
        "Region": details.get("business_card_address_region", ""),
        "Postal Code": details.get("business_card_address_zip", ""),
        "Phone 1": details.get("business_card_phone1", ""),
        "Phone 2": details.get("business_card_phone2", ""),
        "Email": details.get("business_card_email", "")
    }

def main():
    # Step 1: Fetch all members
    members_list = fetch_members_list()
    print(f"Total members in list: {len(members_list)}")

    # Step 2: Load existing data for resume
    detailed_members = load_existing_data()
    print(f"Already have {len(detailed_members)} detailed records. Resuming...")

    # Step 3: Iterate with progress bar
    for member in tqdm(members_list, desc="Fetching details", unit="member"):
        member_id = str(member.get("id"))
        permit_key = member.get("mp")  # Prefer mp from first call
        if not permit_key:
            continue  # Skip if no permit in first call

        if permit_key not in detailed_members:
            details = fetch_member_details(member_id)
            if details:
                final_key = details.get("member_permit", permit_key)
                detailed_members[final_key] = details
                save_data(detailed_members)  # Save after each new record
            time.sleep(0.2)  # Avoid hammering the API

    print(f"Completed! Total detailed records: {len(detailed_members)}")

    # Step 4: Flatten and export to CSV
    flattened_data = [flatten_member(details) for details in detailed_members.values()]
    df = pd.DataFrame(flattened_data)
    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    print(f"CSV file saved as {CSV_FILE}")

if __name__ == "__main__":
    main()