from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Fingerprint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fingerprint_data = db.Column(db.String, nullable=False)

def init_db(app):
    db.init_app(app)

def insert_into_table(fingerprint_data):
    fingerprint = Fingerprint(fingerprint_data=fingerprint_data)
    db.session.add(fingerprint)
    db.session.commit()

def search_in_table(fingerprint_data):
    fingerprint = Fingerprint.query.filter_by(fingerprint_data=fingerprint_data).first()
    if fingerprint:
        return True
    else:
        return False
