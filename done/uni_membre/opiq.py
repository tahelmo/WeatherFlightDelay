
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python opiq.py --nom "Aubé" --prenom "Stéphane"

OPIQ Member Scraper using fixed selectors
- Search page:
    Nom field: id="txt_nom"
    Prénom field: id="txt_prenom"
    Search button: id="cmd_send"
- Results page:
    Member link: id="HLNom1"
- Details page:
    Data container: id="Table2"
"""

import argparse
import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

URL = "https://prive.opiq.qc.ca/WEBPrive/Tableau_acces.aspx"

def build_driver(headless=False):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,900")
    return webdriver.Chrome(options=options)



def extract_details(html):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    data = {}

    # Original table extraction (two-column rows)
    table = soup.find("table", id="Table2")
    if table:
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) == 2:
                key = cells[0].get_text(strip=True)
                val = cells[1].get_text(" ", strip=True)
                if key and val:
                    data[key] = val


    # Add Limitation from LabStatut0 (strip "Limitation:" prefix)
    limitation_span = soup.find("span", id="LabStatut0")
    if limitation_span:
        text = limitation_span.get_text(strip=True)
        if text.lower().startswith("limitation:"):
            text = text[len("Limitation:"):].strip()
        data["Limitation"] = text


    return data



def scrape_member(nom, prenom, headless=False):
    driver = build_driver(headless)
    wait = WebDriverWait(driver, 15)
    try:
        driver.get(URL)

        # Fill search form
        wait.until(EC.presence_of_element_located((By.ID, "txt_nom"))).send_keys(nom)
        wait.until(EC.presence_of_element_located((By.ID, "txt_prenom"))).send_keys(prenom)

        # Click search
        wait.until(EC.element_to_be_clickable((By.ID, "cmd_send"))).click()

        # Click first member link
        wait.until(EC.element_to_be_clickable((By.ID, "HLNom1"))).click()

        # Wait for details page
        wait.until(EC.presence_of_element_located((By.ID, "Table2")))
        html = driver.page_source
        return extract_details(html)

    finally:
        driver.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape OPIQ member details")
    parser.add_argument("--nom", required=True, help="Nom de famille")
    parser.add_argument("--prenom", required=True, help="Prénom")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    args = parser.parse_args()

    details = scrape_member(args.nom, args.prenom, args.headless)
    print(json.dumps(details, ensure_ascii=False, indent=2))
