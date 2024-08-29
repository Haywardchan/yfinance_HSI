import yfinance as yf
from flask import Flask, request, render_template_string, jsonify, make_response
from flask_restful import Api, Resource
import json
from models import Stock, Portfolio, Asset_allocator, Stock_chooser

app = Flask(__name__)
api = Api(app)

class stock_API(Resource):
    def get(self, query):
        with open(f'json_data/{query}.json') as f:
            res = f.read()
        return json.loads(res), 201
    
class cropped_data_API(Resource):
    def get(self, stock_number, start_month, start_year, end_month, end_year):
        # Construct the file path based on the stock number
        filepath = f"json_data/{stock_number}_10y.json"
        
        # Load the data from the JSON file
        with open(filepath) as f:
            data = json.load(f)
        
        # Filter the data based on the start month, start year, end month, and end year
        cropped_data = [
            entry for entry in data 
            if (entry['Date'] >= f"{start_year}-{start_month.zfill(2)}-01" and 
                entry['Date'] <= f"{end_year}-{end_month.zfill(2)}-31")
        ]
        
        return make_response(jsonify(cropped_data), 200)

class optimal_portfolio_API(Resource):
    def get(self, init_price, period):
        model = Asset_allocator(Portfolio()).load_asset_allocator("storage/saved_asset_allocator").portfolio
        prices = model.prices(float(init_price), period)
        return make_response(jsonify(prices.to_dict()), 201)

class cropped_optimal_portfolio_API(Resource):
    def get(self, init_price, start_month, start_year, end_month, end_year):
        model = Asset_allocator(Portfolio()).load_asset_allocator("storage/saved_asset_allocator").portfolio
        prices = model.prices(float(init_price), "10y")
        
        # Filter the prices based on the start month, start year, end month, and end year
        cropped_prices = prices[
            (prices.index >= f"{start_year}-{start_month.zfill(2)}-01") & 
            (prices.index <= f"{end_year}-{end_month.zfill(2)}-31")
        ]
        cropped_prices = cropped_prices * float(init_price) / cropped_prices.iloc[0]
        return make_response(jsonify(cropped_prices.to_dict()), 201)

class rebalanced_portfolio_API(Resource):
    def get(self, init_price, days_to_rebalance, period):
        model = Asset_allocator(Portfolio()).load_asset_allocator("storage/saved_asset_allocator").portfolio
        prices = model.rebalanced_prices(float(init_price), int(days_to_rebalance), period)
        return make_response(jsonify(prices.to_dict()), 201)

class cropped_rebalanced_portfolio_API(Resource):
    def get(self, init_price, days_to_rebalance, start_month, start_year, end_month, end_year):
        model = Asset_allocator(Portfolio()).load_asset_allocator("storage/saved_asset_allocator").portfolio
        prices = model.rebalanced_prices(float(init_price), int(days_to_rebalance), "10y")
        
        # Filter the prices based on the start month, start year, end month, and end year
        cropped_prices = prices[
            (prices.index >= f"{start_year}-{start_month.zfill(2)}-01") & 
            (prices.index <= f"{end_year}-{end_month.zfill(2)}-31")
        ]
        cropped_prices = cropped_prices * float(init_price) / cropped_prices.iloc[0]
        return make_response(jsonify(cropped_prices.to_dict()), 201)

class index_API(Resource):
    def get(self, init_price, index, period):
        stockIndex = ''
        if index =='hsi':
            stockIndex = '^HSI'
        elif index =='sse':
            stockIndex = '000001.SS'
        elif index =='nasdaq':
            stockIndex = '^NDX'
        
        portfolio = Portfolio()
        portfolio.add_stock(Stock(stockIndex, index=index), 1)
        prices = portfolio.prices(float(init_price), period)  # Adjust this line as necessary
        return make_response(jsonify(prices.to_dict()), 201)

class cropped_index_API(Resource):
    def get(self, init_price, index, start_month, start_year, end_month, end_year):
        stockIndex = ''
        if index =='hsi':
            stockIndex = '^HSI'
        elif index =='sse':
            stockIndex = '000001.SS'
        elif index =='nasdaq':
            stockIndex = '^NDX'
        
        portfolio = Portfolio()
        portfolio.add_stock(Stock(stockIndex, index=index), 1)
        prices = portfolio.prices(float(init_price), "10y")
        
        # Filter the prices based on the start month, start year, end month, and end year
        cropped_prices = prices[
            (prices.index >= f"{start_year}-{start_month.zfill(2)}-01") & 
            (prices.index <= f"{end_year}-{end_month.zfill(2)}-31")
        ] 
        cropped_prices = cropped_prices * float(init_price) / cropped_prices.iloc[0]
        
        return make_response(jsonify(cropped_prices.to_dict()), 201)

class genetic_mutation_API(Resource):
    def get(self, num_stocks, num_iterations, index):
        if index =='hsi':
            filepath = '^HSI_1y.csv'
        elif index =='sse':
            filepath = '000001.SS_1y.csv'
        elif index =='nasdaq':
            filepath = '^NDX_1y.csv'
        print(filepath)
        stocks_portfolio = Stock_chooser(target_stocks=int(num_stocks), num_stocks=15, stocks_file=filepath, index=index, mutation_rate=0.3).genetic_algorithm(num_generations=10, sol_per_generation=30, keep_parents=2)
        model = Asset_allocator(stocks_portfolio, generations=int(num_iterations))
        model.run()
        return make_response(jsonify({
            "stocks": list(map(lambda stock: stock.stock_name, model.portfolio.stocks)),
            "weights": list(map(lambda weight: weight * 100, model.portfolio.portions)),
            "risk": model.portfolio.risk,
            "roi": model.portfolio.roi,
            "sharpe_ratio": model.portfolio.sharpe_ratio,
            "var": model.portfolio.var,
            "beta": model.portfolio.beta,
        }), 201)

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import text
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:1234@localhost/finance'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Database_API(Resource):
    def get(self, period, index):
        # Query to fetch data from the dynamically named table based on the period and index
        table_name = f'stock_performance_{index}_{period}'
        results = db.session.execute(text(f"SELECT * FROM {table_name}")).mappings().all()
        print([result for result in results])
        data = []
        for row in results:
            row_dict = dict(row)
            data.append(row_dict)
        return make_response(jsonify(data), 200)

class StockHeadquartersAPI(Resource):
    def get(self, stock_code):
        stock = yf.Ticker(stock_code)
        info = stock.info
        
        headquarters = info.get('address1', 'Headquarters not found for the given stock code.')
        
        return make_response(jsonify({
            "stock_code": stock_code,
            "headquarters": headquarters
        }), 200)

class RefreshDataAPI(Resource):
    def post(self):
        # Logic to refresh the data
        try:
            # Assuming a function refresh_data() exists that handles the data refresh
            from main import refresh
            refresh()
            return make_response(jsonify({"message": "Data refreshed successfully."}), 200)
        except Exception as e:
            return make_response(jsonify({"error": str(e)}), 500)

api.add_resource(RefreshDataAPI, '/api/refresh_data')
api.add_resource(StockHeadquartersAPI, '/api/stock_headquarters/<string:stock_code>')
api.add_resource(Database_API, '/api/database/<string:index>/<string:period>')
api.add_resource(stock_API, '/api/stock/<string:query>')
api.add_resource(optimal_portfolio_API, '/api/optimal_portfolio/<string:init_price>/<string:period>')
api.add_resource(cropped_optimal_portfolio_API, '/api/cropped_optimal_portfolio/<string:init_price>/<string:start_month>/<string:start_year>/<string:end_month>/<string:end_year>')
api.add_resource(rebalanced_portfolio_API, '/api/rebalanced_portfolio/<string:init_price>/<string:days_to_rebalance>/<string:period>')
api.add_resource(cropped_rebalanced_portfolio_API, '/api/cropped_rebalanced_portfolio/<string:init_price>/<string:days_to_rebalance>/<string:start_month>/<string:start_year>/<string:end_month>/<string:end_year>')
api.add_resource(genetic_mutation_API, '/api/genetic_mutation/<string:num_stocks>/<string:num_iterations>/<string:index>')
api.add_resource(index_API, '/api/index/<string:init_price>/<string:index>/<string:period>')
api.add_resource(cropped_data_API, '/api/cropped/<string:stock_number>/<string:start_month>/<string:start_year>/<string:end_month>/<string:end_year>')
api.add_resource(cropped_index_API, '/api/cropped_index/<string:init_price>/<string:index>/<string:start_month>/<string:start_year>/<string:end_month>/<string:end_year>')

if __name__=='__main__':
    app.run(debug=True)
