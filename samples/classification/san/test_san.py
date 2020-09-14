import logging
import os
import time
from itertools import count

import numpy
import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from matplotlib import pyplot
from neodroidvision.classification.architectures.self_attention_network import (
    SelfAttentionTypeEnum,
    make_san,
)
from san_utilities import cal_accuracy, intersection_and_union_gpu

from draugr import AverageMeter
from draugr.torch_utilities import Split

if __name__ == "__main__":

    def main(shuffle: bool = True, how_many_batches=10, batch_size=1):
        def get_logger():
            logger_name = "main-logger"
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(handler)
            return logger

        from samples.classification.san.configs.imagenet_san10_patchwise import (
            SAN_CONFIG,
        )

        dataset = SAN_CONFIG.dataset_type(SAN_CONFIG.dataset_path, Split.Validation)

        logger = get_logger()
        logger.info(SAN_CONFIG)
        logger.info("=> creating model ...")
        logger.info(f"Classes: {dataset.response_shape[0]}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in SAN_CONFIG.test_gpu
        )
        model = make_san(
            self_attention_type=SelfAttentionTypeEnum(SAN_CONFIG.self_attention_type),
            layers=SAN_CONFIG.layers,
            kernels=SAN_CONFIG.kernels,
            num_classes=dataset.response_shape[0],
        )
        logger.info(model)
        model = torch.nn.DataParallel(model.cuda())

        if os.path.isdir(SAN_CONFIG.save_path):
            logger.info(f"=> loading checkpoint '{SAN_CONFIG.model_path}'")
            checkpoint = torch.load(SAN_CONFIG.model_path)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            logger.info(f"=> loaded checkpoint '{SAN_CONFIG.model_path}'")
        else:
            raise RuntimeError(f"=> no checkpoint found at '{SAN_CONFIG.model_path}'")

        criterion = nn.CrossEntropyLoss(ignore_index=SAN_CONFIG.ignore_label)

        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=SAN_CONFIG.test_workers,
            pin_memory=True,
        )

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

        if how_many_batches:
            T = range(how_many_batches)
        else:
            T = count()

        for i, (input, target) in zip(T, val_loader):
            data_time.update(time.time() - end)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with torch.no_grad():
                output = model(input)
                pyplot.imshow(dataset.inverse_base_transform(input[0].cpu()))
                pyplot.title(
                    f"pred:{dataset.category_names[output.max(1)[1][0].item()]} truth:{dataset.category_names[target[0].item()]}"
                )
                pyplot.show()

            loss = criterion(output, target)
            top1, top5 = cal_accuracy(output, target, topk=(1, 5))
            n = input.size(0)
            loss_meter.update(loss.item(), n), top1_meter.update(
                top1.item(), n
            ), top5_meter.update(top5.item(), n)

            intersection, union, target = intersection_and_union_gpu(
                output.max(1)[1],
                target,
                dataset.response_shape[0],
                SAN_CONFIG.ignore_label,
            )
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

            if (i + 1) % SAN_CONFIG.print_freq == 0:
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

        logger.info(
            f"Val result: mIoU/mAcc/allAcc/top1/top5 {mIoU:.4f}/{mAcc:.4f}/{allAcc:.4f}/{top1_meter.avg:.4f}/"
            f"{top5_meter.avg:.4f}."
        )
        for i in range(dataset.response_shape[0]):
            if target_meter.sum[i] > 0:
                logger.info(
                    f"Class_{i} Result: iou/accuracy {iou_class[i]:.4f}/{accuracy_class[i]:.4f} Count:{target_meter.sum[i]}"
                )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        print(loss_meter.avg, mIoU, mAcc, allAcc, top1_meter.avg, top5_meter.avg)

    main()
