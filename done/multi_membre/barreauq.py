import requests
from bs4 import BeautifulSoup
import json
import time

BASE = "https://www.barreau.qc.ca/fr/trouver-un-avocat/resultats/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
}

# --- Parsing de la liste des avocats ---
def parse_barreau_results(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    results = []

    rows = soup.select("tr")
    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 3:
            link_tag = cols[0].select_one("a")
            name_tag = link_tag.select_one("span") if link_tag else None
            name = name_tag.get_text(strip=True) if name_tag else ""
            profile_url = link_tag["href"] if link_tag and link_tag.has_attr("href") else ""

            location_spans = cols[1].find_all("span")
            city = location_spans[0].get_text(strip=True) if len(location_spans) > 0 else ""
            country = location_spans[1].get_text(strip=True) if len(location_spans) > 1 else ""

            employer = cols[2].get_text(strip=True)

            results.append({
                "name": name,
                "city": city,
                "country": country,
                "employer": employer,
                "profile_url": f"https://www.barreau.qc.ca{profile_url}" if profile_url else ""
            })

    return results

# --- Parsing des détails d'un avocat ---
def parse_member_details(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    details = {}

    general_tab = soup.select_one("#generalTab")
    if not general_tab:
        return details

    for dt, dd in zip(general_tab.select("dt"), general_tab.select("dd")):
        label = dt.get_text(strip=True)
        value = dd.get_text(" ", strip=True)

        if "Société" in label or "employeur" in label:
            details["employer_detail"] = value
        elif "Adresse" in label:
            # Supprimer les liens et ne garder que le texte principal
            address_text = dd.get_text(" ", strip=True)
            # Retirer "Itinéraire" et "Voir sur la carte"
            address_text = address_text.replace("Itinéraire", "").replace("Voir sur la carte", "").strip()
            details["address"] = address_text
        elif "Téléphone" in label:
            details.setdefault("phones", []).append(value)
        elif "Télécopieur" in label:
            details["fax"] = value
        elif "Courriel" in label:
            email_tag = dd.select_one("a[href^='mailto:']")
            details["email"] = email_tag.get_text(strip=True) if email_tag else value
        elif "Domaines" in label:
            details["practice_areas"] = value
        elif "Langues" in label:
            details["languages"] = value
        elif "Année" in label:
            details["year_registered"] = value

    return details

# --- Recherche pour une appellation ---
def search_barreau(bn: str, region: str) -> list[dict[str, str]]:
    params = {"bn": bn, "r": region}
    session = requests.Session()
    session.headers.update(HEADERS)

    print(f"[INFO] Recherche des avocats pour '{bn}'...")
    r = session.get(BASE, params=params, timeout=30)
    r.raise_for_status()

    results = parse_barreau_results(r.text)
    print(f"[INFO] {len(results)} avocats trouvés pour '{bn}'.")

    for i, member in enumerate(results, 1):
        if member.get("profile_url"):
            print(f"[INFO] ({i}/{len(results)}) Détails pour {member['name']}...")
            detail_resp = session.get(member["profile_url"], timeout=30)
            detail_resp.raise_for_status()
            member.update(parse_member_details(detail_resp.text))
            time.sleep(1)  # Pause pour éviter surcharge

    return results

# --- Recherche pour plusieurs appellations ---
def search_barreau_multiple(bn_list: list[str], region: str, save_json_path=None):
    all_results = []
    for bn in bn_list:
        results = search_barreau(bn=bn, region=region)
        all_results.extend(results)
        time.sleep(1)

    # Supprimer doublons (clé = name|employer)
    unique_results = {f"{r['name']}|{r['employer']}": r for r in all_results}.values()

    if save_json_path:
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(list(unique_results), f, ensure_ascii=False, indent=2)
        print(f"[INFO] JSON fusionné sauvegardé: {save_json_path}")

    return list(unique_results)

if __name__ == "__main__":
    # Variantes pour CISSSO
    bn_variants = [
        "CISSS de l'Outaouais",
        "CISSSO",
        "Centre intégré de santé et de services sociaux de l'Outaouais",
        "Centre intégré de santé et de services sociaux"
    ]

    data = search_barreau_multiple(
        bn_list=bn_variants,
        region="05",
        save_json_path="barreau_results_all.json"
    )

    for i, row in enumerate(data, 1):
        print(f"{i:02d}. {row['name']} -> {row.get('email', 'Pas de courriel')}")