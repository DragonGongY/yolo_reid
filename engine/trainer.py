# encoding: utf-8
import logging
import os
import time

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from loguru import logger
from utils.reid_metric import R1_mAP, R1_mAP_reranking


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def save_checkpoint(model, optimizer, epoch, output_dir, is_best=False, checkpoint_name='checkpoint'):
    """Save model checkpoint"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if is_best:
        torch.save(checkpoint, os.path.join(output_dir, 'best_model.pth'))
    else:
        torch.save(checkpoint, os.path.join(output_dir, f'{checkpoint_name}_epoch_{epoch}.pth'))


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
        num_query,
        start_epoch,
        args
):
    """
    Main training function without center loss
    """
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger.info("Start training")
    
    # Create evaluator
    evaluator = create_supervised_evaluator(model, 
        metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, 
        device=device)

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        
        # Forward pass
        score, feat = model(img)
        loss = loss_func(score, feat, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()

    trainer = Engine(train_step)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch
        logger.info(f"Starting training from epoch {start_epoch}")

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader), 
                               engine.state.metrics['avg_loss'], scheduler.get_lr()[0]))

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_checkpoint_handler(engine):
        if engine.state.epoch % checkpoint_period == 0:
            save_checkpoint(model, optimizer, engine.state.epoch, output_dir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            logger.info(f"Running validation at epoch {engine.state.epoch}")
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)


def do_train_with_center(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        center_criterion,
        num_query,
        start_epoch,
        args
):
    """
    Training function with center loss
    """
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger.info("Start training with center loss")
    
    # Create evaluator
    evaluator = create_supervised_evaluator(model, 
        metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, 
        device=device)

    def train_step_with_center(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        
        # Forward pass
        score, feat = model(img)
        loss = loss_func(score, feat, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update center loss
        for param in center_criterion.parameters():
            param.grad.data *= (1. / 0.0005)  # center loss weight
        optimizer_center.step()
        
        return loss.item()

    trainer = Engine(train_step_with_center)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch
        logger.info(f"Starting training with center loss from epoch {start_epoch}")

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader), 
                               engine.state.metrics['avg_loss'], scheduler.get_lr()[0]))

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_checkpoint_handler(engine):
        if engine.state.epoch % checkpoint_period == 0:
            save_checkpoint(model, optimizer, engine.state.epoch, output_dir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            logger.info(f"Running validation at epoch {engine.state.epoch}")
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)