from flask import Flask, render_template
import pandas as pd
import numpy as np

# initiate flask
app = Flask(__name__)

# HOME
@app.route('/')
def route_home():
    return render_template('base.htm')

# INDEX
@app.route('/index')
def route_index():
    return render_template('index.htm')

# DATASET
if __name__ =='__main__':
    app.run(debug = True)