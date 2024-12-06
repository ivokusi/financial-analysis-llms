from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from dotenv import load_dotenv
import concurrent.futures
import yfinance as yf
import requests
import json
import os   

class PineconeUtils:

    def __init__(self, index_name: str, namespace: str):

        print("INITIALIZING...")

        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        self.index_name = index_name
        self.namespace = namespace

        load_dotenv()
        
        PineconeVectorStore(index_name=self.index_name, embedding=HuggingFaceEmbeddings())

        self.successful_tickers, self.unsuccessful_tickers = self._load_history()

        print("COMPLETED INITIALIZATION...")

    def get_tickers(self):
        
        """
        Downloads and parses the Stock ticker symbols from the GitHub-hosted SEC company tickers JSON file.

        Returns:
            dict: A dictionary containing company tickers and related information.

        Notes:
            The data is sourced from the official SEC website via a GitHub repository:
            https://raw.githubusercontent.com/team-headstart/Financial-Analysis-and-Automation-with-LLMs/main/company_tickers.json
        """

        # URL to fetch the raw JSON file from GitHub
        url = "https://raw.githubusercontent.com/team-headstart/Financial-Analysis-and-Automation-with-LLMs/main/company_tickers.json"

        # Making a GET request to the URL
        response = requests.get(url)

        # Checking if the request was successful
        if response.status_code == 200:
            
            # Parse the JSON content directly
            company_tickers = json.loads(response.content.decode('utf-8'))

            # Optionally save the content to a local file for future use
            with open("company_tickers.json", "w", encoding="utf-8") as file:
                json.dump(company_tickers, file, indent=4)

            print("FILE DOWNLOADED SUCCESSFULLY AND SAVED AS 'COMPANY_TICKERS.JSON'")
            return company_tickers
        
        else:
        
            print(f"FAILED TO DOWNLOAD FILE. STATUS CODE: {response.status_code}")
            return None

    def _load_history(self):

        # Initialize tracking lists
        
        successful_tickers = []
        unsuccessful_tickers = []

        # Load existing successful/unsuccessful tickers
        
        try:
            
            with open('successful_tickers.txt', 'r') as f:
                successful_tickers = [line.strip() for line in f if line.strip()]
            print(f"LOADED {len(successful_tickers)} SUCCESSFUL TICKERS")

        except FileNotFoundError:
            
            print("NO EXISTING SUCCESSFUL TICKERS FILE FOUND")

        try:
            
            with open('unsuccessful_tickers.txt', 'r') as f:
                unsuccessful_tickers = [line.strip() for line in f if line.strip()]
            print(f"LOADED {len(unsuccessful_tickers)} UNSUCCESSFUL TICKERS")

        except FileNotFoundError:
            
            print("NO EXISTING UNSUCCESSFUL TICKERS FILE FOUND")

        return successful_tickers, unsuccessful_tickers

    def _get_stock_info(self, symbol: str) -> dict:

        """
        Retrieves and formats detailed information about a stock from Yahoo Finance.

        Args:
            symbol (str): The stock ticker symbol to look up.

        Returns:
            dict: A dictionary containing detailed stock information, including ticker, name,
                business summary, city, state, country, industry, sector, market cap, volume,
                PE ratio, price, and analyst recommendation.
        """

        data = yf.Ticker(symbol)
        stock_info = data.info

        properties = {
            "Ticker": stock_info.get("symbol", "Information not available"),
            "Name": stock_info.get("longName", "Information not available"),
            "Business Summary": stock_info.get("longBusinessSummary"),
            "City": stock_info.get("city", "Information not available"),
            "State": stock_info.get("state", "Information not available"),
            "Country": stock_info.get("country", "Information not available"),
            "Industry": stock_info.get("industry", "Information not available"),
            "Sector": stock_info.get("sector", "Information not available"),
            "Market Cap": stock_info.get("marketCap", "Information not available"),
            "Volume": stock_info.get("averageVolume", "Information not available"),
            "PE Ratio": stock_info.get("trailingPE", "Information not available"),
            "Price": stock_info.get("currentPrice", "Information not available"),
            "Analyst Recommendation": stock_info.get("recommendationKey", "Information not available")
        }

        return properties

    def _process_stock(self, stock_ticker: str) -> str:
        
        # Skip if already processed
        
        if stock_ticker in self.successful_tickers:
            return f"Already processed {stock_ticker}"

        try:
            
            # Get and store stock data
            stock_data = self._get_stock_info(stock_ticker)
            stock_description = stock_data['Business Summary']

            # Store stock description in Pinecone
            PineconeVectorStore.from_documents(
                documents=[Document(page_content=stock_description, metadata=stock_data)],
                embedding=HuggingFaceEmbeddings(),
                index_name=self.index_name,
                namespace=self.namespace
            )

            # Track success
            with open('successful_tickers.txt', 'a') as f:
                f.write(f"{stock_ticker}\n")
            
            self.successful_tickers.append(stock_ticker)

            return f"Processed {stock_ticker} successfully"

        except Exception as e:
            
            # Track failure
            with open('unsuccessful_tickers.txt', 'a') as f:
                f.write(f"{stock_ticker}\n")
            
            self.unsuccessful_tickers.append(stock_ticker)

            return f"ERROR processing {stock_ticker}: {e}"

    def parallel_process_stocks(self, tickers: list, max_workers: int = 10) -> None:

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            
            future_to_ticker = {
                executor.submit(self._process_stock, ticker): ticker
                for ticker in tickers
            }

            for future in concurrent.futures.as_completed(future_to_ticker):

                ticker = future_to_ticker[future]

                try:
                    result = future.result()
                    print(result)

                    # Handle errors in the result (if they follow a specific format)
                    if result.startswith("ERROR"):
                        print(f"ERROR ENCOUNTERED IN {ticker}: {result}")

                except Exception as exc:
                    # Log the exception but continue processing other tickers
                    print(f'{ticker} GENERATED AN EXCEPTION: {exc}')

pc_utils = PineconeUtils(index_name="stocks", namespace="stock-descriptions")

tickers = pc_utils.get_tickers()
tickers_to_process = [tickers[num]["ticker"] for num in tickers.keys()]

pc_utils.parallel_process_stocks(tickers_to_process, max_workers=4)