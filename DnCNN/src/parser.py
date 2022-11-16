import argparse


def get_parser(description):
    _parser = argparse.ArgumentParser(description=description)

    _parser.add_argument("--gpu", type=int, default=0)
    _parser.add_argument("--device", type=str, default="cuda:0")

    _parser.add_argument("--num_channels", type=int, default=1)
    _parser.add_argument("--num_features", type=int, default=64)
    _parser.add_argument("--num_layers", type=int, default=17)
    _parser.add_argument("--batch_size", type=int, default=128)

    _parser.add_argument("--weight_decay", type=float, default=0.00001)
    _parser.add_argument("--lr", type=float, default=0.1)
    _parser.add_argument("--momentum", type=float, default=0.9)
    _parser.add_argument("--milestone", type=int, default=20)
    _parser.add_argument("--epoch", type=int, default=60)



    # _args = _parser.parse_args()
    return _parser


if __name__ == "__main__":
    par = get_parser("test")
    # par.
    par.add_argument("--noise_level_max", type=int, default=25)
    print(par.parse_args())