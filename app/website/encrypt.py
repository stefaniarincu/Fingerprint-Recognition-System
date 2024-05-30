import tenseal as ts
import base64
import os
from dotenv import load_dotenv

load_dotenv()

def write_data(file_name, data):
    if type(data) == bytes:
        data = base64.b64encode(data)

    with open(file_name, 'wb') as f:
        f.write(data)

def read_data(file_name):
    with open(file_name, "rb") as f:
        data = f.read()

    return base64.b64decode(data)

def generate_context():
    secret_filename = os.getenv("SECRET_CONTEXT_FILE")
    public_filename = os.getenv("PUBLIC_CONTEXT_FILE")

    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.generate_galois_keys()
    context.global_scale = 2**40

    secret_context = context.serialize(save_secret_key=True)

    if not os.path.exists(secret_filename) or os.path.getsize(secret_filename) == 0:
        write_data(secret_filename, secret_context)
        print("1")

    public_context = context.serialize()

    context.make_context_public() #drop the secret_key from the context
    public_context = context.serialize()

    if not os.path.exists(public_filename) or os.path.getsize(public_filename) == 0:
        write_data(public_filename, public_context)
        print("2")

def ecrypt_fingercode(feature_vector):
    generate_context()

    context = ts.context_from(read_data(os.getenv("SECRET_CONTEXT_FILE")))

    enc_fingercode = ts.ckks_vector(context, feature_vector)
    enc_fingercode_proto = enc_fingercode.serialize()

    return enc_fingercode_proto

def apply_threshold(euclidean_dist, threshold=100):
    if euclidean_dist < 100:
        return 1
    else:
        return 0

def calculate_euclidean_dist(fingercode_1, fingercode_2):
    generate_context()

    context = ts.context_from(read_data(os.getenv("PUBLIC_CONTEXT_FILE")))

    enc_fingercode_1 = ts.lazy_ckks_vector_from(fingercode_1)
    enc_fingercode_1.link_context(context)

    enc_fingercode_2 = ts.lazy_ckks_vector_from(fingercode_2)
    enc_fingercode_2.link_context(context)

    euclidean_dist = enc_fingercode_1 - enc_fingercode_2
    euclidean_dist = euclidean_dist.dot(euclidean_dist)

    context = ts.context_from(read_data(os.getenv("SECRET_CONTEXT_FILE")))
    euclidean_dist.link_context(context)

    euclidean_dist_plain = euclidean_dist.decrypt()[0]

    return euclidean_dist_plain