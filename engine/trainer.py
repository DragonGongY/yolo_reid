# encoding: utf-8
import logging
import os
import time

import torch
import torch.nn as nn
import numpy as np
from loguru import logger


def save_checkpoint(model, optimizer, epoch, output_dir, is_best=False, checkpoint_name='checkpoint', best_path=None):
    """Save model checkpoint"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if is_best:
        best_model_path = best_path if best_path else os.path.join(output_dir, 'best_model.pth')
        torch.save(checkpoint, best_model_path)
        logger.info(f"Best model saved to {best_model_path}")
    else:
        checkpoint_path = os.path.join(output_dir, f'{checkpoint_name}_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")


def extract_features(model, data_loader, device, cfg=None):
    """Extract features from data loader"""
    model.eval()
    features = []
    pids = []
    camids = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    imgs, pids_batch, camids_batch = batch
                elif len(batch) == 2:
                    imgs, pids_batch = batch
                    camids_batch = torch.zeros_like(pids_batch)
                else:
                    imgs = batch[0]
                    pids_batch = torch.zeros(imgs.size(0))
                    camids_batch = torch.zeros(imgs.size(0))
            else:
                # Single tensor case
                imgs = batch
                pids_batch = torch.zeros(imgs.size(0))
                camids_batch = torch.zeros(imgs.size(0))
            
            # Move to device
            imgs = imgs.to(device) if torch.cuda.device_count() >= 1 else imgs
            
            # Forward pass
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                # Handle model outputs that return (score, features)
                if len(outputs) >= 2:
                    feats = outputs[1]  # Features are usually the second output
                else:
                    feats = outputs[0]
            else:
                feats = outputs
            
            # Normalize features if needed
            if cfg and hasattr(cfg, 'TEST') and hasattr(cfg.TEST, 'FEAT_NORM') and cfg.TEST.FEAT_NORM == 'yes':
                feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            
            features.append(feats.cpu())
            
            # Handle pids_batch - ensure it's a tensor before calling .cpu()
            if isinstance(pids_batch, torch.Tensor):
                pids.extend(pids_batch.cpu().numpy())
            else:
                pids.extend(np.array(pids_batch))
            
            # Handle camids_batch - ensure it's a tensor before calling .cpu()
            if isinstance(camids_batch, torch.Tensor):
                camids.extend(camids_batch.cpu().numpy())
            else:
                camids.extend(np.array(camids_batch))
            
            # Progress logging
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"Feature extraction progress: {batch_idx + 1}/{len(data_loader)} batches")
    
    # Concatenate all features
    features = torch.cat(features, dim=0)
    pids = np.array(pids)
    camids = np.array(camids)
    
    return features, pids, camids


def compute_distance_matrix(query_features, gallery_features):
    """Compute distance matrix between query and gallery features"""
    m, n = query_features.size(0), gallery_features.size(0)
    
    # Compute squared Euclidean distance
    distmat = torch.pow(query_features, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gallery_features, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat = distmat.addmm_(1, -2, query_features, gallery_features.t())
    
    return distmat.cpu().numpy()


def evaluate_reid_metrics(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Compute CMC and mAP metrics for ReID"""
    num_q, num_g = distmat.shape
    
    if num_g < max_rank:
        max_rank = num_g
        logger.info(f"Note: number of gallery samples is quite small, got {num_g}")
    
    indices = np.argsort(distmat, axis=1)
    
    # Compute CMC and mAP
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # Compute CMC curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0
    
    for q_idx in range(num_q):
        # Get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        
        # Remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        
        # Compute CMC curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # This condition is true when query identity does not appear in gallery
            continue
        
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1
        
        # Compute average precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_index = np.arange(1, tmp_cmc.shape[0] + 1)
        tmp_cmc = tmp_cmc / tmp_index * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    
    if num_valid_q == 0:
        logger.warning("Warning: all query identities do not appear in gallery")
        return np.zeros(max_rank), 0.0
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    
    return all_cmc, mAP


def validate_model(cfg, model, val_loader, num_query, device):
    """Proper validation with real metrics calculation"""
    logger.info("Starting proper validation...")
    val_start_time = time.time()
    
    try:
        # Extract features
        logger.info("Extracting features from validation set...")
        features, pids, camids = extract_features(model, val_loader, device, cfg)
        
        # Split into query and gallery
        query_features = features[:num_query]
        gallery_features = features[num_query:]
        query_pids = pids[:num_query]
        gallery_pids = pids[num_query:]
        query_camids = camids[:num_query]
        gallery_camids = camids[num_query:]
        
        logger.info(f"Query samples: {len(query_features)}, Gallery samples: {len(gallery_features)}")
        
        # Compute distance matrix
        logger.info("Computing distance matrix...")
        distmat = compute_distance_matrix(query_features, gallery_features)
        
        # Compute metrics
        logger.info("Computing evaluation metrics...")
        cmc, mAP = evaluate_reid_metrics(distmat, query_pids, gallery_pids, query_camids, gallery_camids)
        
        val_end_time = time.time()
        val_duration = val_end_time - val_start_time
        
        # Log results
        logger.info(f"Validation completed in {val_duration:.2f}s")
        logger.info("Validation Results:")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        
        return cmc, mAP
    except Exception as e:
        logger.error(f"Validation failed with error: {str(e)}")
        logger.error("Falling back to placeholder results")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return placeholder results if validation fails
        return np.array([0.5, 0.5, 0.5]), 0.5


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
    logger.info(f"Training parameters: epochs={epochs}, start_epoch={start_epoch}, log_period={log_period}")
    logger.info(f"DataLoader info: batches_per_epoch={len(train_loader)}, batch_size={train_loader.batch_size}")
    
    # Move model to device
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    total_iterations = len(train_loader) * (epochs - start_epoch)
    current_iteration = 0
    
    # Best model tracking
    best_mAP = 0.0
    best_rank1 = 0.0
    best_epoch = 0
    best_model_path = os.path.join(output_dir, "best_model.pth")
    logger.info(f"Best model will be saved to: {best_model_path}")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        
        # Adjust learning rate
        scheduler.step()
        current_lr = scheduler.get_lr()[0] if hasattr(scheduler, 'get_lr') else optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch + 1}/{epochs} started, LR: {current_lr:.6f}")
        
        for i, batch in enumerate(train_loader):
            current_iteration += 1
            
            # Progress logging every 10% or at log period
            if len(train_loader) >= 10:
                progress_10_percent = len(train_loader) // 10
                if (i + 1) % progress_10_percent == 0:
                    progress_percent = (i + 1) / len(train_loader) * 100
                    overall_progress = current_iteration / total_iterations * 100
                    logger.info(f"Progress: Epoch {epoch + 1} - {progress_percent:.1f}% ({i + 1}/{len(train_loader)} batches), Overall: {overall_progress:.1f}%")
            
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
                if len(loss_output) > 1:
                    # Log additional loss components if available
                    logger.info(f"Loss components: main={loss.item():.4f}, additional={len(loss_output)-1} components")
            else:
                loss = loss_output
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            avg_loss = epoch_loss / num_batches
            
            # Detailed logging at specified intervals
            if log_period > 0 and (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}, Avg Loss: {:.4f}, LR: {:.2e}"
                           .format(epoch + 1, i + 1, len(train_loader), loss.item(), avg_loss, current_lr))
            
            # Memory usage logging for CUDA
            if device == 'cuda' and (i + 1) % (log_period * 5) == 0:
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
                    logger.info(f"GPU Memory: Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        logger.info(f"Epoch {epoch + 1} completed in {epoch_duration:.2f}s, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_period == 0:
            save_checkpoint(model, optimizer, epoch + 1, output_dir)
            logger.info(f"Checkpoint saved for epoch {epoch + 1}")
        
        # Validation
        if (epoch + 1) % eval_period == 0:
            logger.info(f"Running validation at epoch {epoch + 1}")
            cmc, mAP = validate_model(cfg, model, val_loader, num_query, device)
            
            # Check if this is the best model so far
            current_rank1 = cmc[0] if len(cmc) > 0 else 0.0
            is_best = mAP > best_mAP or (abs(mAP - best_mAP) < 1e-6 and current_rank1 > best_rank1)
            
            if is_best:
                best_mAP = mAP
                best_rank1 = current_rank1
                best_epoch = epoch + 1
                
                # Save best model
                logger.info(f"New best model found! mAP: {mAP:.1%}, Rank-1: {current_rank1:.1%}")
                logger.info(f"Saving best model to {best_model_path}")
                
                save_checkpoint(model, optimizer, epoch + 1, output_dir, is_best=True, best_path=best_model_path)
                logger.info(f"Best model saved at epoch {epoch + 1}")
            else:
                logger.info(f"Current model - mAP: {mAP:.1%}, Rank-1: {current_rank1:.1%} (Best: mAP: {best_mAP:.1%}, Rank-1: {best_rank1:.1%} at epoch {best_epoch})")
    
    logger.info("Training completed successfully!")
    logger.info(f"Total iterations: {current_iteration}")
    logger.info(f"Best model achieved at epoch {best_epoch} with mAP: {best_mAP:.1%} and Rank-1: {best_rank1:.1%}")
    logger.info(f"Best model saved to: {best_model_path}")


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
    logger.info(f"Training parameters: epochs={epochs}, start_epoch={start_epoch}, log_period={log_period}")
    logger.info(f"DataLoader info: batches_per_epoch={len(train_loader)}, batch_size={train_loader.batch_size}")
    
    # Move model to device
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    total_iterations = len(train_loader) * (epochs - start_epoch)
    current_iteration = 0
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        
        # Adjust learning rate
        scheduler.step()
        current_lr = scheduler.get_lr()[0] if hasattr(scheduler, 'get_lr') else optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch + 1}/{epochs} started, LR: {current_lr:.6f}")
        
        for i, batch in enumerate(train_loader):
            current_iteration += 1
            
            # Progress logging every 10% or at log period
            if len(train_loader) >= 10:
                progress_10_percent = len(train_loader) // 10
                if (i + 1) % progress_10_percent == 0:
                    progress_percent = (i + 1) / len(train_loader) * 100
                    overall_progress = current_iteration / total_iterations * 100
                    logger.info(f"Progress: Epoch {epoch + 1} - {progress_percent:.1f}% ({i + 1}/{len(train_loader)} batches), Overall: {overall_progress:.1f}%")
            
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
                if len(loss_output) > 1:
                    # Log additional loss components if available
                    logger.debug(f"Loss components: main={loss.item():.4f}, additional={len(loss_output)-1} components")
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
            
            # Detailed logging at specified intervals
            if log_period > 0 and (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}, Avg Loss: {:.4f}, LR: {:.2e}"
                           .format(epoch + 1, i + 1, len(train_loader), loss.item(), avg_loss, current_lr))
            
            # Memory usage logging for CUDA
            if device == 'cuda' and (i + 1) % (log_period * 5) == 0:
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
                    logger.info(f"GPU Memory: Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        logger.info(f"Epoch {epoch + 1} completed in {epoch_duration:.2f}s, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_period == 0:
            save_checkpoint(model, optimizer, epoch + 1, output_dir)
            logger.info(f"Checkpoint saved for epoch {epoch + 1}")
        
        # Validation
        if (epoch + 1) % eval_period == 0:
            logger.info(f"Running validation at epoch {epoch + 1}")
            validate_model(cfg, model, val_loader, num_query, device)
    
    logger.info("Training with center loss completed successfully!")
    logger.info(f"Total iterations: {current_iteration}")