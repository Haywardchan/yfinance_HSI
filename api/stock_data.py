from flask import Flask, request, render_template_string
from flask_restful import Api, Resource
import json
app = Flask(__name__)
api = Api(app)

class stock_API(Resource):
    def get(self, query):
        with open(f'json_data/{query}.json') as f:
            res = f.read()
        return json.loads(res), 201
    
api.add_resource(stock_API, '/api/stock/<string:query>')

if __name__=='__main__':
    app.run(debug=True)
