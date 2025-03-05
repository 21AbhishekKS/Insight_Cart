import re
import requests
from bs4 import BeautifulSoup
from .models import Product, AmazonProduct, eBayProduct, SnapdealProduct, AjioProduct

# Constants for the user-agent headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.5"
}

# Selenium imports
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from webdriver_manager.chrome import ChromeDriverManager

# Initialize the Selenium WebDriver for headless browsing
def initialize_driver():
    chrome_options = Options()
    #chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=chrome_options)
    return driver



# Scrape Amazon
import time  # Import the time module

import time  # Import the time module

def scrape_amazon(query):
    search_query = query.replace(' ', '+')
    amazon_url = f"https://www.amazon.in/s?k={search_query}"
    driver = initialize_driver()
    driver.get(amazon_url)

    product_list = []

    try:
        # Wait for the product listings to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, '//div[@data-component-type="s-search-result"]'))
        )

        # Find all product containers using XPath
        products = driver.find_elements(By.XPATH, '//div[@data-component-type="s-search-result"]')

        for product in products[:30]:  # Limit to 30 products
            try:
                # Extract the product title
                try:
                    title = product.find_element(By.XPATH, './/h2').text
                except NoSuchElementException:
                    title = 'No Title Available'

                """try :
                    title2=product.find_element(By.XPATH, './/span').text
                except NoSuchElementException:
                    title2='No Title Available'"""

                # Extract the product link
                try:
                    link = product.find_element(By.XPATH, './/a').get_attribute('href')
                except NoSuchElementException:
                    link = 'No Link Available'

                # Extract the product price
                try:
                    price = product.find_element(By.XPATH, './/span[@class="a-price-whole"]').text
                except NoSuchElementException:
                    price = 'No Price Available'

                # Extract the product rating
                ratings_box = product.find_elements(By.CSS_SELECTOR, '.a-icon-alt')  # Use a CSS selector for ratings
                if ratings_box:  # Check if the ratings element exists
                    rating = ratings_box[0].get_attribute('innerHTML')
                else:
                    rating = 'No Ratings Available'

                # Extract the number of reviews
                try:
                    reviews = product.find_element(By.XPATH, './/span[@class="a-size-base"]').text
                except NoSuchElementException:
                    reviews = 'No Reviews Available'

                # Append the product data to the list
                product_list.append({
                    'title': title,
                    'link': link,
                    'price': price,
                    'rating': rating[:3],  # Extract the numeric part of the rating
                    'reviews': reviews
                })

            except Exception as e:
                print(f"Error extracting product details: {e}")

            # Add a short delay to avoid being flagged as a bot
            time.sleep(1)

        print(f"Amazon: Successfully scraped {len(product_list)} products.")

    except Exception as e:
        print(f"Error while scraping Amazon: {e}")

    finally:
        driver.quit()

    # Save the product list (optional)
    #save_products(product_list, 'Amazon')

    return product_list

# Scrape eBay
def usd_to_inr(usd_price):
    conversion_rate = 85  # Static conversion rate (you can replace it with dynamic API if needed)
    return usd_price * conversion_rate

# Scrape eBay with USD to INR conversion
def scrape_ebay(query):
    search_query = query.replace(' ', '+')
    url = f"https://www.ebay.com/sch/i.html?_nkw={search_query}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    products = soup.find_all('div', class_='s-item__info')
    product_list = []

    for product in products:
        title = product.find('div', class_='s-item__title').text.strip() if product.find('div', class_='s-item__title') else 'No Title Available'
        price = product.find('span', class_='s-item__price').text.strip() if product.find('span', class_='s-item__price') else 'No Price Available'
        
        # Extract the dollar amount from the price string (assuming USD currency format)
        price_in_usd = re.search(r'\$(\d+[\.,]?\d*)', price)
        if price_in_usd:
            price_in_usd = float(price_in_usd.group(1).replace(',', ''))
            price_in_inr = usd_to_inr(price_in_usd)  # Convert USD to INR
        else:
            price_in_inr = 0  # Set default value if price can't be parsed

        link = product.find('a', class_='s-item__link')['href'] if product.find('a', class_='s-item__link') else 'No Link Available'
        rating = product.find('span', class_='s-item__reviews-count').text.strip() if product.find('span', class_='s-item__reviews-count') else 'No Rating Available'

        product_list.append({'title': title, 'price': price_in_inr, 'rating': rating, 'link': link})
        if len(product_list) > 20:
            break

    print(f"eBay: Page scraped successfully. Found {len(products)} products.")
    #save_products(product_list, 'eBay')
    return product_list[2::]


# Scrape Snapdeal
def scrape_snapdeal(query):
    search_query = query.replace(' ', '+')
    url = f"https://www.snapdeal.com/search?keyword={search_query}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    products = soup.find_all('div', class_='product-tuple-listing')
    product_list = []

    for product in products:
        title_elem = product.find('p', class_='product-title')
        title = title_elem.text.strip() if title_elem else 'No Title Available'

        link_elem = product.find('a', class_='dp-widget-link')
        link = link_elem['href'] if link_elem else 'No Link Available'

        price_elem = product.find('span', class_='lfloat product-price')
        price = price_elem.text.strip() if price_elem else 'No Price Available'

        # Extract rating from the `filled-stars` div
        rating_elem = product.find('div', class_='filled-stars')
        if rating_elem:
            style_attr = rating_elem.get('style')
            if style_attr:
                width_percentage = float(style_attr.split('width:')[1].replace('%', '').strip())
                rating = (width_percentage / 100) * 5  # Convert to a 5-star rating scale
            else:
                rating = 'No Rating Available'
        else:
            rating = 'No Rating Available'

        product_list.append({'title': title, 'price': price[3::], 'rating': rating, 'link': link})
    print(f"Snapdeal: Page scraped successfully. Found {len(products)} products.")
    #save_products(product_list, 'Snapdeal')
    return product_list

# Scrape Ajio using Selenium
def scrape_ajio(query):
    search_query = query.replace(' ', '+')
    ajio_url = f"https://www.ajio.com/search/?text={search_query}"
    driver = initialize_driver()
    driver.get(ajio_url)

    products = []
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.items'))
        )

        while len(products) < 30:  # Limit to 20 products
            product_containers = driver.find_elements(By.CSS_SELECTOR, 'div.item')
            for product in product_containers:
                try:
                    title = product.find_element(By.CSS_SELECTOR, '.nameCls').text.strip()
                    price = product.find_element(By.CSS_SELECTOR, '.price').text.strip()
                    link = product.find_element(By.CSS_SELECTOR, 'a').get_attribute('href')
                    try:
                        rating = product.find_element(By.CSS_SELECTOR, '._3I65V').text.strip()
                    except Exception:
                        rating = "No Rating Available"
                    
                    products.append({'title': title, 'price': price, 'rating': rating, 'link': link})

                    

                except Exception as e:
                    print(f"Error extracting product details: {e}")

            

    except Exception as e:
        print(f"Ajio Scraping Failed: {e}")
    finally:
        driver.quit()

    #save_products(products, 'Ajio')
    return products

# Save products in respective databases
"""def save_products(products, source):
    for item in products:
        # Retrieve the rating value and check if it's a valid number
        rating = item.get('rating')

        # Check if the rating is a valid number for non-eBay sources
        if source != 'eBay' and not is_valid_rating(rating):
            # Skip saving the product if the rating is not valid
           
            continue  # Skip this iteration if the rating is not valid

        # Save to the respective model depending on the source
        if source == 'Amazon':
            AmazonProduct.objects.create(
                title=item['title'],
                price=item['price'],
                rating=rating,
                link=item['link']
            )
        elif source == 'eBay':
            eBayProduct.objects.create(
                title=item['title'],
                price=item['price'],
                rating=rating,
                link=item['link']
            )
        elif source == 'Snapdeal':
            SnapdealProduct.objects.create(
                title=item['title'],
                price=item['price'],
                rating=rating,
                link=item['link']
            )
        elif source == 'Ajio':
            AjioProduct.objects.create(
                title=item['title'],
                price=item['price'],
                rating=rating,
                link=item['link']
            )

        # Save the product to the general Product model
        Product.objects.create(
            source=source,
            title=item['title'],
            price=item['price'],
            rating=rating,
            link=item['link']
        )"""

# Helper function to check if the rating is a valid number
def is_valid_rating(rating):
    try:
        # Try converting the rating to a float (this works for both integer and decimal ratings)
        float(rating)
        return True
    except ValueError:
        # If it can't be converted to a float, it's not a valid rating
        return False
