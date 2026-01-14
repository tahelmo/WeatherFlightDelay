import time
import sys
from typing import Dict, Tuple, Optional, List, Any
import requests
from bs4 import BeautifulSoup

BASE = "https://www.oiiaq.org"
SEARCH_PATH = "/public/trouver-une-infirmiere-auxiliaire"
SESSION_INFO_PATH = "/actions/users/session-info"  
TIMEOUT = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,image/apng,*/*;q=0.8,"
              "application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "fr-CA,fr;q=0.9,en-CA;q=0.7,en;q=0.6",
    "Cache-Control": "no-cache",
}

def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s

def fetch_directory_page(session: requests.Session) -> str:
    url = BASE + SEARCH_PATH
    r = session.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text

def extract_form_tokens(html: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (csrf_name, csrf_value, site_form_value)
    If CSRF inputs are not present (async CSRF), returns (None, None, site_form_value).
    """
    soup = BeautifulSoup(html, "html.parser")

    site_form_value = None
    site_form_input = soup.select_one('input[name="site_form"]')
    if site_form_input and site_form_input.has_attr("value"):
        site_form_value = site_form_input["value"]

    
    csrf_input = soup.select_one('input[name="CRAFT_CSRF_TOKEN"]')
    if csrf_input and csrf_input.has_attr("value"):
        return "CRAFT_CSRF_TOKEN", csrf_input["value"], site_form_value

    
    meta_name = soup.find("meta", {"name": "csrf-param"})
    meta_token = soup.find("meta", {"name": "csrf-token"})
    if meta_name and meta_token and meta_name.get("content") and meta_token.get("content"):
        return meta_name["content"], meta_token["content"], site_form_value

    
    return None, None, site_form_value

def fetch_csrf_via_session_info(session: requests.Session) -> Tuple[str, str]:
    """
    Calls Craft CMS session-info endpoint to obtain CSRF token name & value.
    """
    url = BASE + SESSION_INFO_PATH
    r = session.get(url, headers={"X-Requested-With": "XMLHttpRequest"}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "csrf" in data and isinstance(data["csrf"], dict):
        name = data["csrf"].get("name")
        value = data["csrf"].get("value")
        if name and value:
            return name, value
    raise RuntimeError("Could not obtain CSRF token via session-info endpoint.")

def prepare_payload(
    site_form_value: Optional[str],
    csrf_name: str,
    csrf_value: str,
    params: Dict[str, str],
) -> Dict[str, str]:
    """
    Build the x-www-form-urlencoded payload including CSRF, site_form, tab, etc.
    """
    payload = {}
    if csrf_name and csrf_value:
        payload[csrf_name] = csrf_value
    if site_form_value:
        payload["site_form"] = site_form_value

    
    payload.update(params)
    return payload

def post_search(session: requests.Session, payload: Dict[str, str]) -> requests.Response:
    url = BASE + SEARCH_PATH
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": BASE,
        "Referer": BASE + SEARCH_PATH,
    }
    r = session.post(url, data=payload, headers=headers, timeout=TIMEOUT)
    return r

def parse_results(html: str) -> List[Dict[str, str]]:
    """
    Minimal parser: tries table rows first, then card/list blocks.
    If nothing is structured, returns a single item with a short snippet.
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    
    tables = soup.find_all("table")
    if tables:
        table = None
        for t in tables:
            if t.find("tr"):
                table = t
                break
        if table:
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            for row in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if cells:
                    item = { (headers[i] if i < len(headers) else f"col_{i}"): cells[i]
                             for i in range(len(cells)) }
                    results.append(item)

    
    if not results:
        cards = soup.select(".search-result, .result, .member, .card")
        for c in cards:
            text = " ".join(c.stripped_strings)
            if text:
                results.append({"text": text})

    
    if not results:
        snippet = " ".join(soup.stripped_strings)[:400]
        results.append({"snippet": snippet})

    return results

def search_oiiaq(
    tab: str = "code",            
    etab_code: str = "",
    first_name: str = "",
    last_name: str = "",
    permit_number: str = "",
    save_html_path: Optional[str] = None,
    save_json_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Orchestrates the flow and returns a Python list of dicts (printed later),
    without any JSON file writing.
    """
    session = get_session()

    html = fetch_directory_page(session)

    csrf_name, csrf_value, site_form_value = extract_form_tokens(html)
    if csrf_name is None or csrf_value is None:
        csrf_name, csrf_value = fetch_csrf_via_session_info(session)

    if not site_form_value:
        print("[WARN] 'site_form' hidden input not found. Attempting submission without it.", file=sys.stderr)

    params = {"tab": tab}
    if tab == "code":
        params["EtablishmentCode"] = etab_code
        params["FirstName"] = first_name
        params["LastName"] = last_name
        params["PermitNumber"] = permit_number
    elif tab == "name":
        params["FirstName"] = first_name
        params["LastName"] = last_name
        params["PermitNumber"] = ""
        params["EtablishmentCode"] = ""
    elif tab == "permit":
        params["PermitNumber"] = permit_number
        params["FirstName"] = ""
        params["LastName"] = ""
        params["EtablishmentCode"] = ""
    else:
        raise ValueError("tab must be one of: 'code', 'name', 'permit'.")

    payload = prepare_payload(site_form_value, csrf_name, csrf_value, params)

    
    for attempt in range(2):
        r = post_search(session, payload)
        if r.status_code == 200:
            break
        time.sleep(1)
        csrf_name, csrf_value = fetch_csrf_via_session_info(session)
        payload = prepare_payload(site_form_value, csrf_name, csrf_value, params)
    r.raise_for_status()

    
    if save_html_path:
        with open(save_html_path, "w", encoding="utf-8") as f:
            f.write(r.text)

    
    results = parse_results(r.text)

    
    if save_json_path:
        import json
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results

if __name__ == "__main__":
    results = search_oiiaq(
        tab="code",
        etab_code="07-11045218",
        first_name="",
        last_name="",
        permit_number="",
        save_html_path="oiiaq_results.html",
        save_json_path="oiiaq_results.json"
    )
    for i, row in enumerate(results, 1):
        print(f"{i:02d}. {row}")