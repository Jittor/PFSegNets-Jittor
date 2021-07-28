import importlib


def get_net(args, criterion):
    net = get_model(network=args.arch, num_classes=args.dataset_cls.num_classes,
                    criterion=criterion, args=args)
    return net


def get_model(network, num_classes, criterion, args):
    module = network[:network.rfind('.')]
    model = network[(network.rfind('.') + 1):]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    if ((model == 'DeepR101_PF_maxavg_deeply') or (model == 'DeepR50_PF_maxavg_deeply')):
        net = net_func(num_classes=num_classes, criterion=criterion, reduce_dim=args.match_dim,
                       max_pool_size=args.maxpool_size, avgpool_size=args.avgpool_size, edge_points=args.edge_points)
    else:
        net = net_func(num_classes=num_classes, criterion=criterion)
    return net
