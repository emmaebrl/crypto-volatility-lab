from cryptoScraper import CryptoScraper

scraper = CryptoScraper(start_date="2024-01-01")
scraper.get_data(["BTC-USD", "ETH-USD"], output_folder="data")

