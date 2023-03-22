import tinycudann

def get_tcnn_inr(args):
    return tinycudann.NetworkWithInputEncoding(
        n_input_dims=2,
        n_output_dims=4 if args['use features'] else 3,
        encoding_config = args['encoding'],
        network_config = args['network'],
    )