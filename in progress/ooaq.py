"""
OOAQ Permit Number Verifier with Selenium

This script automates OOAQ member verification using Selenium WebDriver.

Requirements:
    pip install selenium webdriver-manager

Usage:
    verifier = OOAQSeleniumVerifier()
    result = verifier.verify_permit("03570")
    print(result)
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import re
from typing import Dict, Optional
import json


class OOAQSeleniumVerifier:
    """Automated OOAQ permit verification using Selenium"""
    
    def __init__(self, headless: bool = True, timeout: int = 20):
        """
        Initialize the verifier.
        
        Args:
            headless: Run browser in headless mode (invisible)
            timeout: Maximum wait time for elements (seconds)
        """
        self.url = "https://portail.ooaq.qc.ca/ThinClient/Public/PR/FR/"
        self.timeout = timeout
        self.headless = headless
        self.driver = None
        
    def _setup_driver(self):
        """Setup Chrome WebDriver with options"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Initialize driver
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
    def format_permit_number(self, permit_number: str) -> str:
        """Format permit number to 5 digits"""
        permit_number = re.sub(r'\D', '', permit_number)
        return permit_number.zfill(5)
    
    def verify_permit(self, permit_number: str) -> Dict:
        """
        Verify an OOAQ permit number.
        
        Args:
            permit_number: The permit number to verify
            
        Returns:
            Dictionary with verification results
        """
        formatted_number = self.format_permit_number(permit_number)
        
        result = {
            'permit_number': formatted_number,
            'original_input': permit_number,
            'valid': False,
            'member_info': {},
            'error': None,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            # Setup driver if not already done
            if self.driver is None:
                self._setup_driver()
            
            print(f"Navigating to OOAQ portal...")
            self.driver.get(self.url)
            
            # Wait for page to load
            wait = WebDriverWait(self.driver, self.timeout)
            
            # Wait for the main content to be visible
            print(f"Waiting for portal to load...")
            time.sleep(3)  # Allow JavaScript to initialize
            
            # Try to find and interact with the search form
            # Note: These selectors need to be updated based on actual portal structure
            try:
                # Look for member number input field
                # Common selectors to try
                selectors = [
                    "input[name*='numero']",
                    "input[name*='member']",
                    "input[id*='numero']",
                    "input[id*='member']",
                    "input[type='text']"
                ]
                
                input_field = None
                for selector in selectors:
                    try:
                        input_field = wait.until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        print(f"Found input field with selector: {selector}")
                        break
                    except TimeoutException:
                        continue
                
                if input_field is None:
                    # Get page source for debugging
                    page_source = self.driver.page_source
                    result['error'] = "Could not locate search input field"
                    result['debug_info'] = {
                        'page_title': self.driver.title,
                        'current_url': self.driver.current_url,
                        'page_source_length': len(page_source)
                    }
                    return result
                
                # Clear and enter permit number
                print(f"Entering permit number: {formatted_number}")
                input_field.clear()
                input_field.send_keys(formatted_number)
                
                # Look for search/submit button
                button_selectors = [
                    "button[type='submit']",
                    "input[type='submit']",
                    "button.search",
                    "button.btn-primary",
                    "*[id*='search']",
                    "*[id*='recherche']"
                ]
                
                search_button = None
                for selector in button_selectors:
                    try:
                        search_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                        print(f"Found search button with selector: {selector}")
                        break
                    except NoSuchElementException:
                        continue
                
                if search_button:
                    search_button.click()
                    print("Search button clicked")
                else:
                    # Try submitting the form directly
                    input_field.submit()
                    print("Form submitted directly")
                
                # Wait for results
                time.sleep(3)
                
                # Try to extract results
                print("Attempting to extract results...")
                
                # Look for result containers
                result_selectors = [
                    ".result",
                    ".member-info",
                    ".search-result",
                    "[class*='result']",
                    "[class*='member']"
                ]
                
                member_data = {}
                
                # Try to find and extract text content
                try:
                    # Get the entire page text after search
                    page_text = self.driver.find_element(By.TAG_NAME, "body").text
                    
                    # Look for common patterns in French
                    name_match = re.search(r'Nom[:\s]+(.+)', page_text, re.IGNORECASE)
                    status_match = re.search(r'Statut[:\s]+(.+)', page_text, re.IGNORECASE)
                    
                    if name_match:
                        member_data['name'] = name_match.group(1).strip()
                    if status_match:
                        member_data['status'] = status_match.group(1).strip()
                    
                    # Check if we found any data
                    if member_data:
                        result['valid'] = True
                        result['member_info'] = member_data
                    else:
                        result['error'] = "No member information found - member may not exist or not be current"
                        result['raw_text'] = page_text[:500]  # First 500 chars for debugging
                    
                except Exception as e:
                    result['error'] = f"Error extracting results: {str(e)}"
                
            except TimeoutException as e:
                result['error'] = f"Timeout waiting for page elements: {str(e)}"
            
        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"
            import traceback
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def main():
    """Main function demonstrating the verifier"""
    print("=" * 70)
    print("OOAQ PERMIT VERIFIER - SELENIUM VERSION")
    print("=" * 70)
    print()
    
    test_permit = "03570"
    
    print(f"Testing with permit number: {test_permit}")
    print("-" * 70)
    print()
    
    # Use context manager to ensure browser closes
    with OOAQSeleniumVerifier(headless=False) as verifier:
        print("Starting verification...")
        print()
        
        result = verifier.verify_permit(test_permit)
        
        print("\n" + "=" * 70)
        print("VERIFICATION RESULT")
        print("=" * 70)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print()
        
        if result['valid']:
            print("✓ Permit is VALID")
            print(f"Member Information:")
            for key, value in result['member_info'].items():
                print(f"  - {key.title()}: {value}")
        else:
            print("✗ Verification failed or permit not found")
            if result['error']:
                print(f"Error: {result['error']}")
    
    print("\n" + "=" * 70)
    print("NOTES")
    print("=" * 70)
    print("""
    - The selectors in this script may need adjustment based on the actual
      portal structure
    - Use headless=False to watch the browser in action for debugging
    - The portal may have anti-bot protection that could require additional
      handling
    - For production use, add proper error handling and retry logic
    """)


if __name__ == "__main__":
    main()