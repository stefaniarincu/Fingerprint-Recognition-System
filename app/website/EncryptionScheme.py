import tenseal as ts
import base64
import os
from dotenv import load_dotenv

class EncryptionScheme:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_context()
        return cls._instance

    def _initialize_context(self):
        load_dotenv()

        self.secret_context_file = os.getenv("SECRET_CONTEXT_FILE")
        self.public_context_file = os.getenv("PUBLIC_CONTEXT_FILE")

        if not os.path.exists(self.secret_context_file) or os.path.getsize(self.secret_context_file) == 0 or not os.path.exists(self.public_context_file) or os.path.getsize(self.public_context_file) == 0:
            self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
            self.context.generate_galois_keys()
            self.context.global_scale = 2**40

            secret_context = self.context.serialize(save_secret_key=True)
            self._write_data(self.secret_context_file, secret_context)

            self.context.make_context_public()
            public_context = self.context.serialize()
            self._write_data(self.public_context_file, public_context)
        else:
            public_context = self._read_data(self.public_context_file)
            self.context = ts.context_from(public_context)

    def _write_data(self, file_name, data):
        if isinstance(data, bytes):
            data = base64.b64encode(data)
        
        with open(file_name, 'wb') as f:
            f.write(data)

    def _read_data(self, file_name):
        with open(file_name, "rb") as f:
            data = f.read()
        
        return base64.b64decode(data)

    def encrypt_fingercode(self, feature_vector):
        enc_fingercode = ts.ckks_vector(self.context, feature_vector)
        enc_fingercode_proto = enc_fingercode.serialize()
        
        return enc_fingercode_proto

    def apply_threshold(self, euclidean_dist, threshold=805000):
        if euclidean_dist < threshold:
            return 1
        else:
            return 0

    def calculate_euclidean_dist(self, fingercode_1, fingercode_2):
        enc_fingercode_1 = ts.lazy_ckks_vector_from(fingercode_1)
        enc_fingercode_1.link_context(self.context)

        enc_fingercode_2 = ts.lazy_ckks_vector_from(fingercode_2)
        enc_fingercode_2.link_context(self.context)

        euclidean_dist = enc_fingercode_1 - enc_fingercode_2
        euclidean_dist = euclidean_dist.dot(euclidean_dist)

        context_secret = ts.context_from(self._read_data(self.secret_context_file))
        euclidean_dist.link_context(context_secret)
        euclidean_dist_plain = euclidean_dist.decrypt()[0]

        return euclidean_dist_plain
