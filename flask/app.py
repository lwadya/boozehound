import sys
import os
from flask import Flask, request, render_template

code_dir = os.path.join(os.path.dirname(sys.path[0]), 'code')
sys.path.append(code_dir)
from lw_pickle import read_pickle
from cocktail_recommender import cocktail_recommender

# Loads model from pickle file
reco_pk = '../data/reco.pk'
cr = read_pickle(reco_pk)

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def index():
    '''
    Gathers input from user and displays results using a Flask HTML template

    Args:
        None

    Returns:
        None
    '''
    if request.method == 'POST':

        # Collects user input and uses it to get recommendations
        search = request.form.get('search')
        weirdness = float(request.form.get('weird'))
        status, df = cr.recommend(search, weirdness=weirdness/100)

        # True status means the recommender returns data
        if status:
            recos = (df[['name', 'image', 'ingredients', 'url']]
                     .to_dict(orient='records'))
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
