from flask import Flask, request, render_template_string, jsonify, make_response
from flask_restful import Api, Resource
import json
from models import Stock, Portfolio, Asset_allocator

app = Flask(__name__)
api = Api(app)

class stock_API(Resource):
    def get(self, query):
        with open(f'json_data/{query}.json') as f:
            res = f.read()
        return json.loads(res), 201
    
class optimal_portfolio_API(Resource):
    def get(self, init_price):
        model = Asset_allocator(Portfolio()).load_asset_allocator("storage/saved_asset_allocator").portfolio
        prices = model.prices(float(init_price))
        return make_response(jsonify(prices.to_dict()), 201)

class rebalanced_portfolio_API(Resource):
    def get(self, init_price, days_to_rebalance):
        model = Asset_allocator(Portfolio()).load_asset_allocator("storage/saved_asset_allocator").portfolio
        prices = model.rebalanced_prices(float(init_price), int(days_to_rebalance))
        return make_response(jsonify(prices.to_dict()), 201)
    
api.add_resource(stock_API, '/api/stock/<string:query>')
api.add_resource(optimal_portfolio_API, '/api/optimal_portfolio/<string:init_price>')
api.add_resource(rebalanced_portfolio_API, '/api/optimal_portfolio/<string:init_price>/<string:days_to_rebalance>')

if __name__=='__main__':
    app.run(debug=True)
