# Flask imports
from flask import Flask
from flask import request, render_template
from random import randint

# Recommender imports
import pickle

from cocktail_recommender import cocktail_recommender

# Load model from pickle
reco_pk = '../data/reco.pk'
with open(reco_pk, 'rb') as f:
    cr = pickle.load(f)

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def user_form():
    if request.method == 'POST':
        search = request.form.get('search')
        weirdness = float(request.form.get('weird'))
        status, df = cr.recommend(search, weirdness=weirdness/100)

        if status:
            recos = (df[['name', 'image', 'ingredients', 'url']]
                     .to_dict(orient='records'))
            '''
            # Format ingredients list
            max_ingr = 5
            for x in recos:
                num_ingr = len(x['ingredients'])
                x['ingredients'] = x['ingredients'][:max_ingr]
                if num_ingr > max_ingr:
                    x['ingredients'][-1] = '...'
            '''
        else:
            recos = None
    else:
        search = recos = weirdness = None

    return render_template('index.html',
                           last_search=search,
                           recos=recos,
                           last_weird=str(weirdness))

if __name__ == '__main__':
    app.run()
