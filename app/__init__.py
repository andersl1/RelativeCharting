from flask import Flask

def create_app():
    app = Flask(__name__)
    app.secret_key = 'aksldfj124nsafja'  # Replace with a unique and secure key

    # Import and register blueprint
    from .routes import main
    app.register_blueprint(main)

    return app