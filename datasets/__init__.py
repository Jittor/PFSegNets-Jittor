"""
Dataset setup and loaders
This file including the different datasets processing pipelines
"""
from datasets import iSAID

import jittor.transform as transforms
import transforms.joint_transforms as joint_transforms
import transforms.transforms as extended_transforms


def setup_loaders(args):
    """
    Setup Data Loaders[Currently supports Cityscapes, Mapillary and ADE20kin]
    input: argument passed by the user
    return:  training data loader, validation data loader loader,  train_set
    """

    if args.dataset == 'iSAID':
        args.dataset_cls = iSAID
        args.train_batch_size = args.bs_mult * args.ngpu
        if args.bs_mult_val > 0:
            args.val_batch_size = args.bs_mult_val * args.ngpu
        else:
            args.val_batch_size = args.bs_mult * args.ngpu

    args.num_workers = 4 * args.ngpu
    if args.test_mode:
        args.num_workers = 1

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Geometric image transformations
    train_joint_transform_list = [
        joint_transforms.RandomSizeAndCrop(args.crop_size,
                                           False,
                                           pre_size=args.pre_size,
                                           scale_min=args.scale_min,
                                           scale_max=args.scale_max,
                                           ignore_index=args.dataset_cls.ignore_label),
        joint_transforms.Resize(args.crop_size),
        joint_transforms.RandomHorizontallyFlip()]

    if args.dataset == 'iSAID':
        train_joint_transform_list = [
            joint_transforms.Resize(args.crop_size),
            joint_transforms.RandomHorizontallyFlip(),
            joint_transforms.RandomVerticalFlip(),
            joint_transforms.RandomRotateThreeDegree()]
        val_joint_transform_list = [
            joint_transforms.Resize(args.crop_size)
        ]

    # Image appearance transformations
    train_input_transform = []
    if args.color_aug:
        train_input_transform += [transforms.ColorJitter(
            brightness=args.color_aug,
            contrast=args.color_aug,
            saturation=args.color_aug,
            hue=args.color_aug)]

    if args.bblur:
        train_input_transform += [extended_transforms.RandomBilateralBlur()]
    elif args.gblur:
        train_input_transform += [extended_transforms.RandomGaussianBlur()]
    else:
        pass

    train_input_transform += [transforms.ToTensor(),
                              transforms.ImageNormalize(*mean_std)]
    train_input_transform = transforms.Compose(train_input_transform)

    val_input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ImageNormalize(*mean_std)
    ])

    target_transform = extended_transforms.MaskToTensor()

    # relax the segmentation border
    if args.jointwtborder:
        target_train_transform = extended_transforms.RelaxedBoundaryLossToTensor(args.dataset_cls.ignore_label,
                                                                                 args.dataset_cls.num_classes)
    else:
        target_train_transform = extended_transforms.MaskToTensor()

    edge_map = args.joint_edge_loss_pfnet

    if args.dataset == 'iSAID':
        train_set = args.dataset_cls.ISAIDDataset(
            args.train_batch_size, True, args.num_workers,
            'semantic', 'train', args.maxSkip,
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            target_transform=target_train_transform,
            dump_images=args.dump_augmentation_images,
            class_uniform_title=args.class_uniform_tile,
            test=args.test_mode,
            cv_split=args.cv,
            scf=args.scf,
            hardnm=args.hardnm,
            edge_map=edge_map,
            thicky=args.thicky)
        val_set = args.dataset_cls.ISAIDDataset(
            args.val_batch_size, False, args.num_workers,
            'semantic', 'val', 0,
            joint_transform_list=val_joint_transform_list,
            transform=val_input_transform,
            target_transform=target_transform,
            test=False,
            cv_split=args.cv,
            scf=None)
    train_set.set_attrs(drop_last=True)
    val_set.set_attrs(drop_last=True)
    
    return train_set, val_set
