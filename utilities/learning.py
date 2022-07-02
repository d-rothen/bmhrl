def adjust_optimizer_lr(optimizer, new_lr):
    for g in optimizer.param_groups:
        g['lr'] = new_lr