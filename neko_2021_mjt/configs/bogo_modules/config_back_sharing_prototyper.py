from neko_2021_mjt.bogo_modules.prototype_gen3 import PrototyperGen3d, PrototyperGen3


def config_prototyper_gen3d(sp_proto, backbone, cam, drop=None, capacity=512, force_proto_shape=None):
    return {
        "bogo_mod": PrototyperGen3d,
        "args":
            {
                "capacity": capacity,
                "sp_proto": sp_proto,
                "backbone": backbone,
                "cam": cam,
                "drop": drop,
                "force_proto_shape": force_proto_shape,
            }
    }


def config_prototyper_gen3(sp_proto, backbone, cam, drop=None, capacity=512, force_proto_shape=None):
    return {
        "bogo_mod": PrototyperGen3,
        "args":
            {
                "capacity": capacity,
                "sp_proto": sp_proto,
                "backbone": backbone,
                "cam": cam,
                "drop": drop,
                "force_proto_shape": force_proto_shape,
            }
    }
