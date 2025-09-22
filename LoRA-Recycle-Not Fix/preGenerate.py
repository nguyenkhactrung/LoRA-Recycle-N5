# This file shows hot to invert images from a pre-trained ViTs with LoRA.
import math
import os
import shutil
import torch
from torch.utils.data import DataLoader
from dataset.cifar100 import Cifar100_Specific
from dataset.cub import CUB_Specific
from dataset.flower import flower_Specific
from dataset.miniimagenet import MiniImageNet_Specific
from lora import LoRA_clipModel
from synthesis import InversionSyntheiszer
from tool import bias, get_model, get_transform, Normalizer, NORMALIZE_DICT, Timer, dataset_classnum, \
    label_abs2relative, bias_end


def prepare_model_for_pre_generate(args, load=True):
    model = get_model(args, load=load)
    model = LoRA_clipModel(model, r=args.rank, num_classes=dataset_classnum[args.dataset])
    model = model.to(args.device)
    return model

def pre_generate(args):
    timer=Timer()
    teacher=prepare_model_for_pre_generate(args,load=True)
    pre_datapool_path = args.pre_datapool_path
    if os.path.exists(pre_datapool_path):
        shutil.rmtree(pre_datapool_path)
        print('remove')
    os.makedirs(pre_datapool_path, exist_ok=True)

    if "32" in args.backbone:
        synthesizer = InversionSyntheiszer(args=args, teacher=None,
                                     img_size=(3, args.resolution, args.resolution),
                                     iterations=2000, lr_g=0.25,
                                     synthesis_batch_size=None,
                                     adv=0.0, bn=0.01, oh=1.0, tv=0.0, l2=0.0, patch_size=32,
                                     save_dir=pre_datapool_path,
                                     transform=get_transform(args, dataset=args.dataset),
                                     normalizer=Normalizer(**NORMALIZE_DICT[args.dataset]),
                                     device=args.device, num_classes=list(range(bias[args.dataset],bias_end[args.dataset]+1)), c_abs_list=None,
                                     max_batch_per_class=10000000)
    elif "16" in args.backbone:
        synthesizer = InversionSyntheiszer(args=args, teacher=None,
                                         img_size=(3, args.resolution, args.resolution),
                                         iterations=2000, lr_g=0.25,
                                         synthesis_batch_size=None,
                                         adv=0.0, bn=0.01, oh=1.0, tv=0.0, l2=0.0,patch_size=16,
                                         save_dir=pre_datapool_path,
                                         transform=get_transform(args, dataset=args.dataset),
                                         normalizer=Normalizer(**NORMALIZE_DICT[args.dataset]),
                                         device=args.device, num_classes=list(range(bias[args.dataset],bias_end[args.dataset]+1)), c_abs_list=None,
                                         max_batch_per_class=10000000)

    #lora training
    specific=list(range(dataset_classnum[args.dataset]))
    if args.dataset=='cifar100':
        trainset = Cifar100_Specific(setname='meta_train', specific=specific, augment=False, resolution=224, mode='train')
        testset = Cifar100_Specific(setname='meta_train', specific=specific, augment=False, resolution=224, mode='test')
    elif args.dataset=='cub':
        trainset = CUB_Specific(setname='meta_train', specific=specific, augment=False, resolution=224,mode='train')
        testset = CUB_Specific(setname='meta_train', specific=specific, augment=False, resolution=224, mode='test')
    elif args.dataset=='flower':
        trainset = flower_Specific(setname='meta_train', specific=specific, augment=False, resolution=224,mode='train')
        testset = flower_Specific(setname='meta_train', specific=specific, augment=False, resolution=224, mode='test')
    elif args.dataset=='miniimagenet':
        trainset = MiniImageNet_Specific(setname='meta_train', specific=specific, augment=False, resolution=224, mode='train')
        testset = MiniImageNet_Specific(setname='meta_train', specific=specific, augment=False, resolution=224, mode='test')
    else:
        raise NotImplementedError

    train_loader = DataLoader(dataset=trainset,
                              num_workers=8,
                              batch_size=64,
                              shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(dataset=testset,
                             num_workers=8,
                             batch_size=64,
                             shuffle=True,
                             pin_memory=True)
    optimizer = torch.optim.Adam(params=teacher.parameters(), lr=0.001)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 50, 80], gamma=0.2)
    num_epoch = 1
    best_acc = None
    for epoch in range(num_epoch):
        # train
        teacher.train()
        for batch_count, batch in enumerate(train_loader):
            optimizer.zero_grad()
            image, abs_label, label_text = batch[0].cuda(args.device), batch[1].cuda(args.device), batch[2]
            relative_label = label_abs2relative(specific=specific, label_abs=abs_label).cuda(args.device)

            logits = teacher.get_image_logits(image)
            criteria = torch.nn.CrossEntropyLoss()
            loss = criteria(logits, relative_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 50)
            optimizer.step()
        lr_schedule.step()
        correct, total = 0, 0
        teacher.eval()
        for batch_count, batch in enumerate(test_loader):
            image, abs_label = batch[0].cuda(args.device), batch[1].cuda(args.device)
            relative_label = label_abs2relative(specific=specific, label_abs=abs_label).cuda(args.device)
            logits = teacher.get_image_logits(image)
            prediction = torch.max(logits, 1)[1]
            correct = correct + (prediction.cpu() == relative_label.cpu()).sum()
            total = total + len(relative_label)
        test_acc = 100 * correct / total
        print(test_acc)
        if best_acc == None or best_acc < test_acc:
            best_acc = test_acc
            best_epoch = epoch
            not_increase = 0
        else:
            not_increase = not_increase + 1
            if not_increase == 60:
                print('early stop at:', best_epoch)
                break

    synthesizer.teacher = teacher

    #preGenerate
    instance_per_class=10
    epoch=40//instance_per_class
    classes_per=4
    lora=int(math.ceil(dataset_classnum[args.dataset]/classes_per))
    for epoch_id in range(epoch):
        for lora_id in range(lora):
            specific=[]
            start=(lora_id*classes_per)%dataset_classnum[args.dataset]
            for delta in range(classes_per):
                specific.append((start+delta)%dataset_classnum[args.dataset])
            print(specific)
            synthesizer.c_abs_list = [i + bias[args.dataset] for i in specific]
            support_query_tensor, _ = synthesizer.synthesize(
                targets=torch.LongTensor(specific * (instance_per_class)),
                student=None, c_num=len(synthesizer.c_abs_list), add=True)
            print('ETA:{}/{}'.format(
                timer.measure(),
                timer.measure(((lora_id+1)*(epoch_id+1)) / (lora*epoch)))
            )