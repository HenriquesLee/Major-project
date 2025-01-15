from flask import Blueprint, render_template

docs_bp = Blueprint('docs', __name__)

@docs_bp.route('/')
def index():
    return render_template('docs.html') 