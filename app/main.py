import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from app.routes import bp

load_dotenv()


def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(bp, url_prefix="/api")
    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, port=port)
