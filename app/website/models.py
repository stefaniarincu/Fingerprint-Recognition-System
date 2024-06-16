from . import db

class Fingerprint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fingercode = db.Column(db.LargeBinary)
