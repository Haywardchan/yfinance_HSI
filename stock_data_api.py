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
    
class optimal_portfolio_API(Resource):
    def get(self, init_price, period):
        model = Asset_allocator(Portfolio()).load_asset_allocator("storage/saved_asset_allocator").portfolio
        prices = model.prices(float(init_price), period)
        return make_response(jsonify(prices.to_dict()), 201)

class rebalanced_portfolio_API(Resource):
    def get(self, init_price, days_to_rebalance, period):
        model = Asset_allocator(Portfolio()).load_asset_allocator("storage/saved_asset_allocator").portfolio
        prices = model.rebalanced_prices(float(init_price), int(days_to_rebalance), period)
        return make_response(jsonify(prices.to_dict()), 201)
    
class genetic_mutation_API(Resource):
    def get(self, num_stocks, num_iterations, index):
        if index =='hsi':
            filepath = '^HSI_1y.csv'
        elif index =='sse':
            filepath = '^000001.SS_1y.csv'
        elif index =='nasdaq':
            filepath = '^NDX_1y.csv'
        portfolio = Stock_chooser(target_stocks=int(num_stocks), num_stocks=int(num_stocks), stocks_file=filepath, mutation_rate=0.3).genetic_algorithm(num_generations=10, sol_per_generation=30, keep_parents=2)
        model = Asset_allocator(portfolio, generations=int(num_iterations))
        model.run()
        return make_response(jsonify({portfolio: str(model.portfolio)}), 201)

api.add_resource(stock_API, '/api/stock/<string:query>')
api.add_resource(optimal_portfolio_API, '/api/optimal_portfolio/<string:init_price>/<string:period>')
api.add_resource(rebalanced_portfolio_API, '/api/optimal_portfolio/<string:init_price>/<string:days_to_rebalance>/<string:period>')
api.add_resource(genetic_mutation_API, '/api/genetic_mutation/<string:num_stocks>/<string:num_iterations>/<string:index>')

if __name__=='__main__':
    app.run(debug=True)
