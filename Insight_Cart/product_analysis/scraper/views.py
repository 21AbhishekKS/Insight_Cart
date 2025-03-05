from django.shortcuts import render
from .utils import scrape_ebay, scrape_snapdeal, scrape_amazon, scrape_ajio
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Function to rank products using clustering and calculate a predicted score
def rank_products_ml(products):
    try:
        df = pd.DataFrame(products)

        # Clean the 'price' column
        df['price'] = df['price'].replace({'\u20b9': '', ',': ''}, regex=True).replace('', np.nan)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')  # Convert to numeric, invalid entries become NaN

        # Drop rows where 'price' is NaN or non-numeric
        df.dropna(subset=['price'], inplace=True)

        # Clean the 'rating' column
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')  # Convert to numeric, invalid entries become NaN
        df['rating'].fillna(0, inplace=True)  # Replace NaN ratings with 0

        # Normalize 'price' and 'rating'
        if not df.empty:
            scaler = MinMaxScaler()
            df[['normalized_price', 'normalized_rating']] = scaler.fit_transform(df[['price', 'rating']])
        else:
            return pd.DataFrame()

        # Clustering with K-Means (only if there's enough data)
        if len(df) >= 3:
            X = df[['normalized_price', 'normalized_rating']].values
            kmeans = KMeans(n_clusters=3, random_state=42)
            df['cluster'] = kmeans.fit_predict(X)
            # Compute distances to centroid
            df['distance_to_centroid'] = np.linalg.norm(X - kmeans.cluster_centers_[df['cluster']], axis=1)
            df['predicted_score'] = (1 / (1 + df['distance_to_centroid'])) * df['rating']
            df = df.sort_values(by='predicted_score', ascending=False)
        return df
    except Exception as e:
        print(f"Error in rank_products_ml: {e}")
        return pd.DataFrame()

# Function to visualize price data
import re  # For regex cleaning

def visualize_price_data(prices, labels=None, future_steps=10):
    try:
        # Clean price data: Remove '₹', commas, and ensure numeric conversion
        cleaned_prices = []
        for price in prices:
            if isinstance(price, str):  # Ensure it's a string before cleaning
                cleaned_price = re.sub(r'[^\d.]', '', price)  # Remove non-numeric characters except '.'
                try:
                    cleaned_prices.append(float(cleaned_price))  # Convert to float
                except ValueError:
                    continue  # Skip invalid prices
            else:
                cleaned_prices.append(float(price))  # Append already numeric values

        if not cleaned_prices:  # Check if prices list is empty after cleaning
            raise ValueError("No valid prices found for visualization.")

        df = pd.DataFrame({'price': cleaned_prices})
        df['label'] = labels if labels else range(len(df))

        # Train Linear Regression model
        model = LinearRegression()
        X = df['label'].values.reshape(-1, 1)
        y = df['price'].values
        model.fit(X, y)

        # Predict future prices
        future_labels = np.arange(len(df), len(df) + future_steps).reshape(-1, 1)
        future_prices = model.predict(future_labels)

        # Plot the graph
        plt.figure(figsize=(10, 6))
        plt.plot(df['label'], df['price'] ,marker='o' ,label='Prices', color='blue')
        plt.plot(future_labels, future_prices, label='Predicted Prices', linestyle='--', color='red')
        plt.xlabel('Index')
        plt.ylabel('Price')
        plt.title('Price Prediction')
        plt.legend()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        graph = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()

        return graph, future_prices.tolist()
    except Exception as e:
        print(f"Error in visualize_price_data: {e}")
        return None, []


# Main view function
def scrape_products(request):
    query = request.GET.get('query', '')

    if not query:
        return render(request, 'scraper/product_list.html', {'message': 'No query provided'})

    try:
        amazon_products = []
        ajio_products = []
        ebay_products = []
        snapdeal_products = []

        # Scrape products from each platform with error handling
        try:
            amazon_products = scrape_amazon(query)
        except Exception as e:
            print(f"Error scraping Amazon: {e}")
        
        try:
            ajio_products = scrape_ajio(query)
        except Exception as e:
            print(f"Error scraping Ajio: {e}")
        
        try:
            ebay_products = scrape_ebay(query)
        except Exception as e:
            print(f"Error scraping eBay: {e}")
        
        try:
            snapdeal_products = scrape_snapdeal(query)
        except Exception as e:
            print(f"Error scraping Snapdeal: {e}")

        # Rank products using ML
        try:
            top_3_amazon = rank_products_ml(amazon_products).head(3)
            top_3_ajio = rank_products_ml(ajio_products).head(3)
            top_3_ebay = rank_products_ml(ebay_products).head(3)
            top_3_snapdeal = rank_products_ml(snapdeal_products).head(3)
        except Exception as e:
            print(f"Error ranking products: {e}")
            top_3_amazon = top_3_ajio = top_3_ebay = top_3_snapdeal = pd.DataFrame()

        # Generate graphs and future prices
        try:
            graph_amazon, future_amazon_prices = visualize_price_data([p['price'] for p in amazon_products if p['price']])
        except:
            graph_amazon, future_amazon_prices = None, []

        try:
            graph_ebay, future_ebay_prices = visualize_price_data([p['price'] for p in ebay_products if p['price']])
        except:
            graph_ebay, future_ebay_prices = None, []
        
        try:
            graph_ajio, future_ajio_prices = visualize_price_data([p['price'] for p in ajio_products if p['price']])
        except:
            graph_ajio, future_ajio_prices = None, []

        try:
            graph_snapdeal, future_snapdeal_prices = visualize_price_data([p['price'] for p in snapdeal_products if p['price']])
        except:
            graph_snapdeal, future_snapdeal_prices = None, []

        # Prepare the context
        context = {
            'amazon_products': amazon_products,
            'ajio_products': ajio_products,
            'ebay_products': ebay_products,
            'snapdeal_products': snapdeal_products,
            'top_3_amazon': top_3_amazon.to_dict(orient='records') if not top_3_amazon.empty else [],
            'top_3_ajio': top_3_ajio.to_dict(orient='records') if not top_3_ajio.empty else [],
            'top_3_ebay': top_3_ebay.to_dict(orient='records') if not top_3_ebay.empty else [],
            'top_3_snapdeal': top_3_snapdeal.to_dict(orient='records') if not top_3_snapdeal.empty else [],
            'graph_amazon': graph_amazon,
            'graph_ebay': graph_ebay,
            'graph_ajio':graph_ajio,
            'graph_snapdeal':graph_snapdeal,
            'future_ajio_prices':future_ajio_prices,
            'future_snapdeal_prices':future_snapdeal_prices,
            'future_amazon_prices': future_amazon_prices,
            'future_ebay_prices': future_ebay_prices,
        }

        return render(request, 'scraper/product_list.html', context)
    except Exception as e:
        print(f"Unhandled error: {e}")
        return render(request, 'scraper/product_list.html', {'message': 'An error occurred while processing your request.'})
