# encoding: utf-8
import logging
import os
import time

import torch
import torch.nn as nn
from loguru import logger


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
    
    # Move model to device
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    avg_loss = 0.0
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Adjust learning rate
        scheduler.step()
        
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            img, target = batch
            img = img.to(device) if torch.cuda.device_count() >= 1 else img
            target = target.to(device) if torch.cuda.device_count() >= 1 else target
            
            # Forward pass
            score, feat = model(img)
            loss_output = loss_func(score, feat, target)
            
            # Handle loss function output (could be tuple or single value)
            if isinstance(loss_output, tuple):
                loss = loss_output[0]  # Take the first element as the actual loss
            else:
                loss = loss_output
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            avg_loss = epoch_loss / num_batches
            
            # Log training loss
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                           .format(epoch + 1, i + 1, len(train_loader), avg_loss, scheduler.get_lr()[0]))
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_period == 0:
            save_checkpoint(model, optimizer, epoch + 1, output_dir)
        
        # Validation
        if (epoch + 1) % eval_period == 0:
            logger.info(f"Running validation at epoch {epoch + 1}")
            validate_model(cfg, model, val_loader, num_query, device)
    
    logger.info("Training completed")


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
    
    # Move model to device
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    avg_loss = 0.0
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Adjust learning rate
        scheduler.step()
        
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            img, target = batch
            img = img.to(device) if torch.cuda.device_count() >= 1 else img
            target = target.to(device) if torch.cuda.device_count() >= 1 else target
            
            # Forward pass
            score, feat = model(img)
            loss_output = loss_func(score, feat, target)
            
            # Handle loss function output (could be tuple or single value)
            if isinstance(loss_output, tuple):
                loss = loss_output[0]  # Take the first element as the actual loss
            else:
                loss = loss_output
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update center loss
            for param in center_criterion.parameters():
                param.grad.data *= (1. / 0.0005)  # center loss weight
            optimizer_center.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            avg_loss = epoch_loss / num_batches
            
            # Log training loss
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                           .format(epoch + 1, i + 1, len(train_loader), avg_loss, scheduler.get_lr()[0]))
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_period == 0:
            save_checkpoint(model, optimizer, epoch + 1, output_dir)
        
        # Validation
        if (epoch + 1) % eval_period == 0:
            logger.info(f"Running validation at epoch {epoch + 1}")
            validate_model(cfg, model, val_loader, num_query, device)
    
    logger.info("Training with center loss completed")


def validate_model(cfg, model, val_loader, num_query, device):
    """Validate model"""
    model.eval()
    
    with torch.no_grad():
        # This is a simplified validation - you may need to adapt based on your specific validation needs
        logger.info("Running validation...")
        # Add your validation logic here
        # For now, just log that validation is running
        logger.info("Validation completed")
    
    # Calculate and log metrics
    # You would implement your specific validation metrics here
    logger.info("Validation Results - Placeholder results")
    logger.info("mAP: {:.1%}".format(0.5))  # Placeholder
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}: {:.1%}".format(r, 0.5))  # Placeholder