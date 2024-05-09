import tenseal as ts
import numpy as np
import base64

def write_data(file_name, data):
    if type(data) == bytes:
        data = base64.b64encode(data)

    with open(file_name, 'wb') as f:
        f.write(data)

def read_data(file_name):
    with open(file_name, "rb") as f:
        data = f.read()
    return base64.b64decode(data)

def enc_feature_vector(feature_vector):
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree = 8192, coeff_mod_bit_sizes = [60, 40, 40, 60])
    context.generate_galois_keys()
    context.global_scale = 2**40

    secret_context = context.serialize(save_secret_key = True)
    write_data("secret.txt", secret_context)

    context.make_context_public() #drop the secret_key from the context
    public_context = context.serialize()
    write_data("public.txt", public_context)

    context = ts.context_from(read_data("secret.txt"))

    enc_feature_vector = ts.ckks_vector(context, feature_vector)

    enc_feature_vector_proto = enc_feature_vector.serialize()

    write_data("enc_FV.txt", enc_feature_vector_proto)

    return enc_feature_vector_proto



'''
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree = 8192, coeff_mod_bit_sizes = [60, 40, 40, 60])
    context.generate_galois_keys()
    context.global_scale = 2**40

    secret_context = context.serialize(save_secret_key = True)
    write_data("secret.txt", secret_context)

    context.make_context_public() #drop the secret_key from the context
    public_context = context.serialize()
    write_data("public.txt", public_context)

    context = ts.context_from(read_data("secret.txt"))

    enc_v1 = ts.ckks_vector(context, img1_embedding)
    enc_v2 = ts.ckks_vector(context, img2_embedding)

    enc_v1_proto = enc_v1.serialize()
    enc_v2_proto = enc_v2.serialize()

    write_data("enc_v1.txt", enc_v1_proto)
    write_data("enc_v2.txt", enc_v2_proto)

    #cloud system will have the public key
    context = ts.context_from(read_data("public.txt"))

    #restore the embedding of person 1
    enc_v1_proto = read_data("enc_v1.txt")
    enc_v1 = ts.lazy_ckks_vector_from(enc_v1_proto)
    enc_v1.link_context(context)

    #restore the embedding of person 2
    enc_v2_proto = read_data("enc_v2.txt")
    enc_v2 = ts.lazy_ckks_vector_from(enc_v2_proto)
    enc_v2.link_context(context)

    #euclidean distance
    euclidean_squared = enc_v1 - enc_v2
    euclidean_squared = euclidean_squared.dot(euclidean_squared)

    #store the homomorphic encrypted squared euclidean distance
    write_data("euclidean_squared.txt", euclidean_squared.serialize())

    try:
        euclidean_squared.decrypt()
    except Exception as err:
        print("Exception: ", str(err))

        #client has the secret key
    context = ts.context_from(read_data("secret.txt"))

    #load euclidean squared value
    euclidean_squared_proto = read_data("euclidean_squared.txt")
    euclidean_squared = ts.lazy_ckks_vector_from(euclidean_squared_proto)
    euclidean_squared.link_context(context)

    #decrypt it
    euclidean_squared_plain = euclidean_squared.decrypt()[0]

    if euclidean_squared_plain < 100:
        print("they are same person")
    else:
        print("they are different persons")
'''
