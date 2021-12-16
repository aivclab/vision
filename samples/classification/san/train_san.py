import logging
import numpy
import os
import random
import shutil
import time
import torch
from draugr import AverageMeter, find_unclaimed_port
from draugr.numpy_utilities import SplitEnum
from draugr.torch_utilities import TensorBoardPytorchWriter
from pathlib import Path
from torch import distributed, multiprocessing, nn
from torch.backends import cudnn
from torch.optim import lr_scheduler

from neodroidvision.classification.architectures.self_attention_network import (
    SelfAttentionTypeEnum,
    make_san,
)
from san_utilities import (
    cal_accuracy,
    intersection_and_union_gpu,
    mixup_data,
    mixup_loss,
    smooth_loss,
)


def get_logger():
    """

    Returns:

    """
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    """

    Args:
      worker_id:
    """
    random.seed(CONFIG.manual_seed + worker_id)


def is_main_process():
    """

    Returns:

    """
    return not CONFIG.multiprocessing_distributed or (
        CONFIG.multiprocessing_distributed and CONFIG.rank % CONFIG.ngpus_per_node == 0
    )


def main_worker(gpu, ngpus_per_node, config):
    """

    Args:
      gpu:
      ngpus_per_node:
      config:
    """
    global CONFIG, best_acc1
    CONFIG, best_acc1 = config, 0
    train_set = config.dataset_type(CONFIG.dataset_path, SplitEnum.training)
    val_set = config.dataset_type(CONFIG.dataset_path, SplitEnum.validation)

    if CONFIG.distributed:
        if CONFIG.dist_url == "env://" and CONFIG.rank == -1:
            CONFIG.rank = int(os.environ["RANK"])
        if CONFIG.multiprocessing_distributed:
            CONFIG.rank = CONFIG.rank * ngpus_per_node + gpu
        distributed.init_process_group(
            backend=CONFIG.dist_backend,
            init_method=CONFIG.dist_url,
            world_size=CONFIG.world_size,
            rank=CONFIG.rank,
        )

    model = make_san(
        self_attention_type=SelfAttentionTypeEnum(CONFIG.self_attention_type),
        layers=CONFIG.layers,
        kernels=CONFIG.kernels,
        num_classes=train_set.response_shape[0],
    )
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG.ignore_label)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CONFIG.base_lr,
        momentum=CONFIG.momentum,
        weight_decay=CONFIG.weight_decay,
    )
    if CONFIG.scheduler == "step":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=CONFIG.step_epochs, gamma=0.1
        )
    elif CONFIG.scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG.epochs)

    if is_main_process():
        global logger, writer
        logger = get_logger()
        writer = TensorBoardPytorchWriter(str(CONFIG.save_path))
        logger.info(CONFIG)
        logger.info("=> creating model ...")
        logger.info(f"Classes: {train_set.response_shape[0]}")
        logger.info(model)
    if CONFIG.distributed:
        torch.cuda.set_device(gpu)
        CONFIG.batch_size = int(CONFIG.batch_size / ngpus_per_node)
        CONFIG.batch_size_val = int(CONFIG.batch_size_val / ngpus_per_node)
        CONFIG.workers = int((CONFIG.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[gpu]
        )
    else:
        model = torch.nn.DataParallel(model.cuda())

    if CONFIG.weight:
        if Path(CONFIG.weight).is_file():
            if is_main_process():
                global logger
                logger.info(f"=> loading weight '{CONFIG.weight}'")
            checkpoint = torch.load(CONFIG.weight)
            model.load_state_dict(checkpoint["state_dict"])
            if is_main_process():
                global logger
                logger.info(f"=> loaded weight '{CONFIG.weight}'")
        else:
            if is_main_process():
                global logger
                logger.info(f"=> no weight found at '{CONFIG.weight}'")

    if CONFIG.resume:
        if Path(CONFIG.resume).is_file():
            if is_main_process():
                global logger
                logger.info(f"=> loading checkpoint '{CONFIG.resume}'")
            checkpoint = torch.load(
                CONFIG.resume, map_location=lambda storage, loc: storage.cuda(gpu)
            )
            CONFIG.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["top1_val"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if is_main_process():
                global logger
                logger.info(
                    f"=> loaded checkpoint '{CONFIG.resume}' (epoch {checkpoint['epoch']})"
                )
        else:
            if is_main_process():
                global logger
                logger.info(f"=> no checkpoint found at '{CONFIG.resume}'")

    if CONFIG.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=CONFIG.batch_size,
        shuffle=(train_sampler is None),
        num_workers=CONFIG.workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=CONFIG.batch_size_val,
        shuffle=False,
        num_workers=CONFIG.workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    for epoch in range(CONFIG.start_epoch, CONFIG.epochs):
        if CONFIG.distributed:
            train_sampler.set_epoch(epoch)
        (
            loss_train,
            mIoU_train,
            mAcc_train,
            allAcc_train,
            top1_train,
            top5_train,
        ) = train(train_loader, model, criterion, optimizer, epoch)
        loss_val, mIoU_val, mAcc_val, allAcc_val, top1_val, top5_val = validate(
            val_loader, model, criterion
        )
        scheduler.step()
        epoch_log = epoch + 1
        if is_main_process():
            global writer
            writer.scalar("loss_train", loss_train, epoch_log)
            writer.scalar("mIoU_train", mIoU_train, epoch_log)
            writer.scalar("mAcc_train", mAcc_train, epoch_log)
            writer.scalar("allAcc_train", allAcc_train, epoch_log)
            writer.scalar("top1_train", top1_train, epoch_log)
            writer.scalar("top5_train", top5_train, epoch_log)
            writer.scalar("loss_val", loss_val, epoch_log)
            writer.scalar("mIoU_val", mIoU_val, epoch_log)
            writer.scalar("mAcc_val", mAcc_val, epoch_log)
            writer.scalar("allAcc_val", allAcc_val, epoch_log)
            writer.scalar("top1_val", top1_val, epoch_log)
            writer.scalar("top5_val", top5_val, epoch_log)

        if (epoch_log % CONFIG.save_freq == 0) and is_main_process():
            filename = CONFIG.save_path / "train_epoch_" + str(epoch_log) + ".pth"
            global logger
            logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {
                    "epoch": epoch_log,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "top1_val": top1_val,
                    "top5_val": top5_val,
                },
                filename,
            )
            if top1_val > best_acc1:
                best_acc1 = top1_val
                shutil.copyfile(filename, CONFIG.save_path / "model_best.pth")
            if epoch_log / CONFIG.save_freq > 2:
                deletename = (
                    CONFIG.save_path
                    / f"train_epoch_{str(epoch_log - CONFIG.save_freq * 2)}.pth"
                )
                os.remove(deletename)


def train(train_loader, model, criterion, optimizer, epoch):
    """

    Args:
      train_loader:
      model:
      criterion:
      optimizer:
      epoch:

    Returns:

    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = CONFIG.epochs * len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if CONFIG.mixup_alpha:
            eps = CONFIG.label_smoothing if CONFIG.label_smoothing else 0.0
            input, target_a, target_b, lam = mixup_data(
                input, target, CONFIG.mixup_alpha
            )
            output = model(input)
            loss = mixup_loss(output, target_a, target_b, lam, eps)
        else:
            output = model(input)
            loss = (
                smooth_loss(output, target, CONFIG.label_smoothing)
                if CONFIG.label_smoothing
                else criterion(output, target)
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        top1, top5 = cal_accuracy(output, target, topk=(1, 5))
        n = input.size(0)
        if CONFIG.multiprocessing_distributed:
            with torch.no_grad():
                loss, top1, top5 = loss.detach() * n, top1 * n, top5 * n
                count = target.new_tensor([n], dtype=torch.long)
                distributed.all_reduce(loss)
                distributed.all_reduce(top1)
                distributed.all_reduce(top5)
                distributed.all_reduce(count)
                n = count.item()
                loss, top1, top5 = loss / n, top1 / n, top5 / n
        loss_meter.update(loss.item(), n), top1_meter.update(
            top1.item(), n
        ), top5_meter.update(top5.item(), n)

        output = output.max(1)[1]
        intersection, union, target = intersection_and_union_gpu(
            output, target, train_loader.dataset.response_shape[0], CONFIG.ignore_label
        )
        if CONFIG.multiprocessing_distributed:
            distributed.all_reduce(intersection)
            distributed.all_reduce(union)
            distributed.all_reduce(target)
        intersection, union, target = (
            intersection.cpu().numpy(),
            union.cpu().numpy(),
            target.cpu().numpy(),
        )
        intersection_meter.update(intersection), union_meter.update(
            union
        ), target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = f"{int(t_h):02d}:{int(t_m):02d}:{int(t_s):02d}"

        if ((i + 1) % CONFIG.print_freq == 0) and is_main_process():
            logger.info(
                f"Epoch: [{epoch + 1}/{CONFIG.epochs}][{i + 1}/{len(train_loader)}] Data {data_time.val:.3f} ("
                f"{data_time.avg:.3f}) Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) Remain {remain_time} Loss "
                f"{loss_meter.val:.4f} Accuracy {accuracy:.4f} Acc@1 {top1_meter.val:.3f} ({top1_meter.avg:.3f}) "
                f"Acc@5 {top5_meter.val:.3f} ({top5_meter.avg:.3f})."
            )
        if is_main_process():
            writer.scalar("loss_train_batch", loss_meter.val, current_iter)
            writer.scalar(
                "mIoU_train_batch",
                numpy.mean(intersection / (union + 1e-10)),
                current_iter,
            )
            writer.scalar(
                "mAcc_train_batch",
                numpy.mean(intersection / (target + 1e-10)),
                current_iter,
            )
            writer.scalar("allAcc_train_batch", accuracy, current_iter)
            writer.scalar("top1_train_batch", top1, current_iter)
            writer.scalar("top5_train_batch", top5, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = numpy.mean(iou_class)
    mAcc = numpy.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if is_main_process():
        logger.info(
            f"Train result at epoch [{epoch + 1}/{CONFIG.epochs}]: mIoU/mAcc/allAcc/top1/top5 {mIoU:.4f}/"
            f"{mAcc:.4f}/{allAcc:.4f}/{top1_meter.avg:.4f}/{top5_meter.avg:.4f}."
        )
    return loss_meter.avg, mIoU, mAcc, allAcc, top1_meter.avg, top5_meter.avg


def validate(val_loader, model, criterion):
    """

    Args:
      val_loader:
      model:
      criterion:

    Returns:

    """
    if is_main_process():
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        loss = criterion(output, target)

        top1, top5 = cal_accuracy(output, target, topk=(1, 5))
        n = input.size(0)
        if CONFIG.multiprocessing_distributed:
            with torch.no_grad():
                loss, top1, top5 = loss.detach() * n, top1 * n, top5 * n
                count = target.new_tensor([n], dtype=torch.long)
                distributed.all_reduce(loss), distributed.all_reduce(
                    top1
                ), distributed.all_reduce(top5), distributed.all_reduce(count)
                n = count.item()
                loss, top1, top5 = loss / n, top1 / n, top5 / n
        loss_meter.update(loss.item(), n), top1_meter.update(
            top1.item(), n
        ), top5_meter.update(top5.item(), n)

        output = output.max(1)[1]
        intersection, union, target = intersection_and_union_gpu(
            output, target, val_loader.dataset.response_shape[0], CONFIG.ignore_label
        )
        if CONFIG.multiprocessing_distributed:
            distributed.all_reduce(intersection), distributed.all_reduce(
                union
            ), distributed.all_reduce(target)
        intersection, union, target = (
            intersection.cpu().numpy(),
            union.cpu().numpy(),
            target.cpu().numpy(),
        )
        intersection_meter.update(intersection), union_meter.update(
            union
        ), target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i + 1) % CONFIG.print_freq == 0) and is_main_process():
            logger.info(
                f"Test: [{i + 1}/{len(val_loader)}] Data {data_time.val:.3f} ({data_time.avg:.3f}) Batch "
                f"{batch_time.val:.3f} ({batch_time.avg:.3f}) Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                f"Accuracy {accuracy:.4f} Acc@1 {top1_meter.val:.3f} ({top1_meter.avg:.3f}) Acc@5 "
                f"{top5_meter.val:.3f} ({top5_meter.avg:.3f})."
            )

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = numpy.mean(iou_class)
    mAcc = numpy.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if is_main_process():
        logger.info(
            f"Val result: mIoU/mAcc/allAcc/top1/top5 {mIoU:.4f}/{mAcc:.4f}/{allAcc:.4f}/{top1_meter.avg:.4f}/"
            f"{top5_meter.avg:.4f}."
        )
        for i in range(val_loader.dataset.response_shape[0]):
            if target_meter.sum[i] > 0:
                logger.info(
                    f"Class_{i} Result: iou/accuracy {iou_class[i]:.4f}/{accuracy_class[i]:.4f} Count:"
                    f"{target_meter.sum[i]}"
                )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
    return loss_meter.avg, mIoU, mAcc, allAcc, top1_meter.avg, top5_meter.avg


if __name__ == "__main__":

    def main():
        """ """
        from samples.classification.san.configs.imagenet_san10_pairwise import (
            SAN_CONFIG,
        )

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in SAN_CONFIG.train_gpu
        )
        if SAN_CONFIG.manual_seed is not None:
            random.seed(SAN_CONFIG.manual_seed)
            numpy.random.seed(SAN_CONFIG.manual_seed)
            torch.manual_seed(SAN_CONFIG.manualSeed)
            torch.cuda.manual_seed(SAN_CONFIG.manualSeed)
            torch.cuda.manual_seed_all(SAN_CONFIG.manualSeed)
            cudnn.benchmark = False
            cudnn.deterministic = True
        if SAN_CONFIG.dist_url == "env://" and SAN_CONFIG.world_size == -1:
            SAN_CONFIG.world_size = int(os.environ["WORLD_SIZE"])
        SAN_CONFIG.distributed = (
            SAN_CONFIG.world_size > 1 or SAN_CONFIG.multiprocessing_distributed
        )
        SAN_CONFIG.ngpus_per_node = len(SAN_CONFIG.train_gpu)
        if len(SAN_CONFIG.train_gpu) == 1:
            SAN_CONFIG.sync_bn = False
            SAN_CONFIG.distributed = False
            SAN_CONFIG.multiprocessing_distributed = False
        if SAN_CONFIG.multiprocessing_distributed:
            port = find_unclaimed_port()
            SAN_CONFIG.dist_url = f"tcp://127.0.0.1:{port}"
            SAN_CONFIG.world_size *= SAN_CONFIG.ngpus_per_node
            multiprocessing.spawn(
                main_worker,
                nprocs=SAN_CONFIG.ngpus_per_node,
                args=(SAN_CONFIG.ngpus_per_node, SAN_CONFIG),
            )
        else:
            main_worker(SAN_CONFIG.train_gpu, SAN_CONFIG.ngpus_per_node, SAN_CONFIG)

    main()
