
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

python oppq.py --permis 92109



OPPQ - Recherche de membre par numéro de permis
URL: https://oppq.connexence.com/ext/oppq/tm/repertoire/trouverMembre.zul?locale=fr

Sélecteurs fournis (XPaths absolus) :
- Champ "Numéro de permis" :
  /html/body/div/div/div/div[5]/div[4]/div/div/div[1]/div[3]/input
- Bouton "Rechercher" :
  /html/body/div/div/div/div[5]/div[4]/div/div/div[1]/div[4]/button
- Container des résultats :
  /html/body/div/div/div/div[5]/div[3]/div/div[2]/div/div/div

Usage :
  python oppq.py --permis 12345
  (optionnel) --headless pour ne pas ouvrir la fenêtre du navigateur
"""

import argparse
import json
import sys
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


URL = "https://oppq.connexence.com/ext/oppq/tm/repertoire/trouverMembre.zul?locale=fr"

XPATH_INPUT = "/html/body/div/div/div/div[5]/div[4]/div/div/div[1]/div[3]/input"
XPATH_BUTTON = "/html/body/div/div/div/div[5]/div[4]/div/div/div[1]/div[4]/button"
XPATH_CONTAINER = "/html/body/div/div/div/div[5]/div[3]/div/div[2]/div/div/div"


def build_driver(headless: bool = False):
    """Construit un driver Chrome avec options utiles."""
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,900")
    options.add_argument("--lang=fr-FR,fr")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")
    # Si Edge est préféré, on peut basculer ici. Pour l’instant, on utilise Chrome.
    return webdriver.Chrome(options=options)





def parse_label_values_from_text(text: str):
    """
    Transforme le texte brut en dictionnaire structuré.
    Gère les blocs d'adresse en trois champs distincts :
    - Adresse (rue)
    - Ville
    - Code postal
    """
    data = {}
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    i = 0
    while i < len(lines):
        key = lines[i]
        if i + 1 < len(lines):
            val = lines[i + 1]
            i += 2

            # Si c'est l'adresse principale, on prend les 3 lignes suivantes
            if key == "Adresse de l'emploi principal":
                # Rue
                rue = val
                ville = ""
                code_postal = ""
                if i < len(lines):
                    ville = lines[i]
                    i += 1
                if i < len(lines):
                    code_postal = lines[i]
                    i += 1
                data["Adresse (rue)"] = rue
                data["Adresse (ville)"] = ville
                data["Adresse (code postal)"] = code_postal
            else:
                data[key] = val
        else:
            i += 1



    return data



def scrape_oppq_by_permit(permit_number: str, headless: bool = False):
    """Recherche par numéro de permis et retourne un dictionnaire des infos extraites."""
    driver = build_driver(headless=headless)
    wait = WebDriverWait(driver, 20)

    try:
        driver.get(URL)

        # Saisir le numéro de permis
        permit_input = wait.until(EC.presence_of_element_located((By.XPATH, XPATH_INPUT)))
        permit_input.clear()
        permit_input.send_keys(permit_number)

        # Cliquer sur Rechercher
        search_btn = wait.until(EC.element_to_be_clickable((By.XPATH, XPATH_BUTTON)))
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", search_btn)
        search_btn.click()

                
        # Attendre que le container existe
        results_div = wait.until(EC.presence_of_element_located((By.XPATH, XPATH_CONTAINER)))

        # Attendre que le texte apparaisse (au moins "Nom" ou "Numéro de permis")
        WebDriverWait(driver, 20).until(
            EC.text_to_be_present_in_element((By.XPATH, XPATH_CONTAINER), "Nom")
        )

        # Extraire le texte
        text = results_div.text.strip()
        # print(f"[DEBUG] Texte brut récupéré: {text}")  # Pour vérifier

        if not text:
            return {"status": "no_results", "message": "Aucun texte dans le container de résultats."}

        # Transformer en dict
        data = parse_label_values_from_text(text)
        data["Numéro de permis (recherché)"] = permit_number

        return data

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        driver.quit()


def main():
    parser = argparse.ArgumentParser(description="OPPQ – Recherche par numéro de permis")
    parser.add_argument("--permis", required=True, help="Numéro de permis à rechercher (ex.: 12345)")
    parser.add_argument("--headless", action="store_true", help="Exécuter sans interface graphique")
    args = parser.parse_args()

    result = scrape_oppq_by_permit(args.permis, headless=args.headless)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
