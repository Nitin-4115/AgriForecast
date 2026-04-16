import time
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

APP_URL = "http://localhost:8501"

# ==========================================
# 1. PYTEST FIXTURE (Browser Setup/Teardown)
# ==========================================
@pytest.fixture(scope="module")
def driver():
    """Initializes the Chrome driver once for the whole test suite."""
    print("\n--- Starting Bulletproof Data-Driven Test Suite ---")
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless') # Uncomment this later if you want tests to run silently in the background
    service = Service(ChromeDriverManager().install())
    browser = webdriver.Chrome(service=service, options=options)
    
    yield browser
    
    time.sleep(2)
    browser.quit()
    print("\n--- Test Suite Complete ---")

# ==========================================
# 2. BULLETPROOF SELENIUM HELPERS
# ==========================================
def set_selectbox(driver, index, value):
    """Uses exact-text matching to prevent React autocomplete errors."""
    dropdown = driver.find_elements(By.CSS_SELECTOR, '[data-baseweb="select"]')[index]
    driver.execute_script("arguments[0].click();", dropdown)
    time.sleep(1) 
    
    input_field = dropdown.find_element(By.TAG_NAME, "input")
    input_field.send_keys(value)
    time.sleep(1.5) 
    
    options = driver.find_elements(By.CSS_SELECTOR, 'li[role="option"]')
    for opt in options:
        if opt.text == value:
            driver.execute_script("arguments[0].click();", opt)
            break
    time.sleep(1.5)

def check_option_exists(driver, dropdown_index, search_term):
    """Types a term into a dropdown to bypass React virtualized DOM limits."""
    dropdown = driver.find_elements(By.CSS_SELECTOR, '[data-baseweb="select"]')[dropdown_index]
    driver.execute_script("arguments[0].click();", dropdown)
    time.sleep(1)

    input_field = dropdown.find_element(By.TAG_NAME, "input")
    # Clear the input field safely
    input_field.send_keys(Keys.CONTROL + "a")
    input_field.send_keys(Keys.BACKSPACE)
    # Search for our target
    input_field.send_keys(search_term)
    time.sleep(1.5)

    options = driver.find_elements(By.CSS_SELECTOR, 'li[role="option"]')
    texts = [opt.text for opt in options]
    
    # Close dropdown by hitting escape
    input_field.send_keys(Keys.ESCAPE)
    time.sleep(0.5)
    
    return search_term in texts

def set_number_input(driver, index, value):
    """Clears and sets Streamlit number inputs."""
    num_inputs = driver.find_elements(By.CSS_SELECTOR, 'input[type="number"]')
    driver.execute_script("arguments[0].click();", num_inputs[index])
    time.sleep(0.5)
    
    num_inputs[index].send_keys(Keys.CONTROL + "a")
    num_inputs[index].send_keys(Keys.DELETE)
    num_inputs[index].send_keys(str(value))
    num_inputs[index].send_keys(Keys.ENTER)
    time.sleep(0.5)

# ==========================================
# 3. TEST DATA CONFIGURATION
# ==========================================
FORECAST_SCENARIOS = [
    ("India", "Potatoes", 2026, 25, 1200, 5000, "SAFE"),
    ("India", "Potatoes", 2026, 45, 100, 5000, "CRITICAL"),
    ("Canada", "Maize", 2029, 10, 150, 8000, "SAFE"),
    ("Brazil", "Soybeans", 2030, 28, 4500, 4000, "CRITICAL"),
    ("France", "Wheat", 2026, 22, 800, 0, "SAFE"),
    ("India", "Potatoes", 2035, 50, 5000, 100000, "CRITICAL"),
]

# ==========================================
# 4. PYTEST TEST CASES
# ==========================================

@pytest.mark.parametrize("area, crop, year, temp, rain, pest, expected_risk", FORECAST_SCENARIOS)
def test_forecast_scenarios(driver, area, crop, year, temp, rain, pest, expected_risk):
    """Tests various climate and input scenarios against the ML models."""
    driver.get(APP_URL)
    wait = WebDriverWait(driver, 10)
    wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Run Forecast')]")))
    time.sleep(1)

    set_selectbox(driver, 0, area)
    set_selectbox(driver, 1, crop)
    
    set_number_input(driver, 0, year)
    set_number_input(driver, 1, temp)
    set_number_input(driver, 2, rain)
    set_number_input(driver, 3, pest)
    
    run_button = driver.find_element(By.XPATH, "//button[contains(., 'Run Forecast')]")
    driver.execute_script("arguments[0].click();", run_button)
    
    time.sleep(3) # Wait for ML inference
    
    # Streamlit uses "stMetricLabel" reliably.
    labels = driver.find_elements(By.CSS_SELECTOR, '[data-testid="stMetricLabel"]')
    
    # labels[1] is the "Failure Probability" label. "./.." gets its parent container.
    risk_card = labels[1].find_element(By.XPATH, "./..")
    risk_result_text = risk_card.text
    
    assert expected_risk in risk_result_text, f"Expected risk {expected_risk}, but got {risk_result_text}"


def test_dynamic_dropdown_filtering(driver):
    """Ensures the UI correctly filters available crops based on region."""
    driver.get(APP_URL)
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '[data-baseweb="select"]')))
    time.sleep(1)
    
    set_selectbox(driver, 0, "Canada")
    
    has_bananas = check_option_exists(driver, 1, "Bananas")
    has_maize = check_option_exists(driver, 1, "Maize")
    
    assert has_bananas is False, "Tropical crop found in cold region!"
    assert has_maize is True, "Expected crop missing from filtered list."


def test_analytics_tab_rendering(driver):
    """Verifies the diagnostic dashboard loads all visualizations."""
    driver.get(APP_URL)
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'button[data-baseweb="tab"]')))
    time.sleep(1)
    
    tabs = driver.find_elements(By.CSS_SELECTOR, 'button[data-baseweb="tab"]')
    driver.execute_script("arguments[0].click();", tabs[1]) # Click Tab 2
    
    time.sleep(3) # Give Streamlit time to load the heavy SVGs
    
    images = driver.find_elements(By.TAG_NAME, 'img')
    assert len(images) >= 20, f"Expected ~22 diagnostic images, only found {len(images)}"