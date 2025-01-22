import requests
import pandas as pd
from datetime import datetime

def get_stock_quote(symbol, api_key):
    """
    Fetch stock quote data, process it into a cleaner format.
    Handles Saudi stocks (.SR) with appropriate currency and formatting.

    Args:
        symbol (str): Stock symbol (e.g., "2222.SR")
        api_key (str): API key for authentication

    Returns:
        pandas.DataFrame: Processed stock quote data
    """
    url = "https://api.bavest.co/v0/quote"

    headers = {
        "accept": "application/json",
        "x-api-key": api_key
    }

    payload = {
        "symbol": symbol
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for bad status codes

        # Get the response data
        data = response.json()

        # Force SAR currency for Saudi stocks
        is_saudi_stock = symbol.upper().endswith('.SR')
        currency = 'SAR' if is_saudi_stock else data['currency']

        # Create a more structured format
        processed_data = {
            "Current Price Data": {
                "Current_Price": data['c'],
                "Daily_Change": data['d'],
                "Daily_Change_Percent": data['dp'],
                "Day_High": data['h'],
                "Day_Low": data['l'],
                "Opening_Price": data['o'],
                "Previous_Close": data['pc']
            },
            "Trading_Info": {
                "Volume": data['v'],
                "Timestamp": datetime.fromtimestamp(data['t']).strftime('%Y-%m-%d %H:%M:%S')
            },
            "Historical_Price": {
                "MA_50_Days": data['historical_price']['50days'],
                "MA_200_Days": data['historical_price']['200days']
            },
            "Company_Metrics": {
                "Market_Cap": data['metrics']['marketCapitalization'],
                "Average_Volume": data['metrics']['avgVolume'],
                "EPS": data['metrics']['eps'],
                "PE_Ratio": data['metrics']['pe/ratio'],
                "Shares_Outstanding": data['metrics']['sharesOutstanding']
            },
            "Additional_Info": {
                "Next_Earnings_Date": data['earningsAnnouncement'],
                "Currency": currency,  # Use determined currency
                "Market": "Saudi Stock Exchange (Tadawul)" if is_saudi_stock else "Other"
            }
        }

        # Convert nested dictionary to flat DataFrame
        flat_data = {}
        for category, values in processed_data.items():
            for key, value in values.items():
                flat_data[f"{category}_{key}"] = [value]

        # Create DataFrame
        df = pd.DataFrame(flat_data)
        print(df)
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {str(e)}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error processing data: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    api_key = "vhJd9c0AYZ8L8uRphQc4p416sBK6ezQ19IQiww6E"
    symbol = "2222.SR"
    quote_data = get_stock_quote(symbol, api_key)