from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, ValDataset, inverse_normalize
from model.contextual_rstart_cnn_vgg16 import RStarCNNVGG16
from torch.utils import data as data_
from contextual_rstar_cnn_trainer import RStarCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, rstar_cnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = rstar_cnn.predict(imgs, gt_bboxes_, [
                                                                     sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    import pickle
    result = {
        "pred_bboxes" : pred_bboxes,
        "pred_labels" : pred_labels, 
        "pred_scores" : pred_scores,
        "gt_bboxes"   : gt_bboxes,
        "gt_labels"   : gt_labels
    }

    with open('results_updated.pkl', 'wb') as f:
        pickle.dump(result, f)

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=False)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    valset = ValDataset(opt)
    val_dataloader = data_.DataLoader(valset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False,
                                       pin_memory=True
                                       )
    # testset = TestDataset(opt)
    # test_dataloader = data_.DataLoader(testset,
    #                                    batch_size=1,
    #                                    num_workers=opt.test_num_workers,
    #                                    shuffle=False, \
    #                                    pin_memory=True
    #                                    )
    rstar_cnn = RStarCNNVGG16()
    print('model construct completed')
#     print(rstar_cnn)
    trainer = RStarCNNTrainer(rstar_cnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    #trainer.vis.text(dataset.db.label_names, win='labels')
    print(dataset.db.label_names)
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        print(f"train its [{len(dataloader)}]")
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)

            # if (ii + 1) % opt.plot_every == 0:
            #     if os.path.exists(opt.debug_file):
            #         ipdb.set_trace()

            #     # plot loss
            #     trainer.vis.plot_many(trainer.get_meter_data())

            #     # plot groud truth bboxes
            #     ori_img_ = inverse_normalize(at.tonumpy(img[0]))
            #     gt_img = visdom_bbox(ori_img_,
            #                          at.tonumpy(bbox_[0]),
            #                          at.tonumpy(label_[0]))
            #     trainer.vis.img('gt_img', gt_img)

            #     # plot predicti bboxes
            #     _bboxes, _labels, _scores = trainer.rstar_cnn.predict([ori_img_], visualize=True)
            #     pred_img = visdom_bbox(ori_img_,
            #                            at.tonumpy(_bboxes[0]),
            #                            at.tonumpy(_labels[0]).reshape(-1),
            #                            at.tonumpy(_scores[0]))
            #     trainer.vis.img('pred_img', pred_img)

            #     # rpn confusion matrix(meter)
            #     trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
            #     # roi confusion matrix
            #     trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
#             log_info = 'lr:{}, loss:{}'.format(str(trainer.rstar_cnn.optimizer.param_groups[0]['lr']),
#                                                       str(trainer.get_meter_data()))
#             print(log_info)
        print(f"val its [{len(val_dataloader)}]")
        eval_result = eval(val_dataloader, rstar_cnn, test_num=opt.test_num)
        # trainer.vis.plot('test_map', eval_result['map'])
        
        # print(f"test its [{len(test_dataloader)}]")
        # test_eval_result = eval(test_dataloader, rstar_cnn, test_num=opt.test_num)
        
        # trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.rstar_cnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, val_map:{}, loss:{}'.format(str(lr_),
                                                      str(eval_result['map']),
                                                      str(trainer.get_meter_data()))
        # trainer.vis.log(log_info)
        print(log_info)
        print(str(eval_result['ap']))

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        else:
            _ = trainer.save(best_map=eval_result['map'])
        if epoch == 5:
            trainer.load(best_path)
            trainer.rstar_cnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        # if epoch == 13: 
        #     break


if __name__ == '__main__':
    import fire

    fire.Fire()
