import torch

ckpt = torch.load('./work_dirs/unet_catdog_sanity/epoch_50.pth', weights_only=False)


weights_only_ckpt = {
    'state_dict': ckpt['state_dict']

}


torch.save(weights_only_ckpt, './work_dirs/unet_catdog_sanity/epoch_50_weights_only.pth')

print("Чистые веса сохранены в epoch_50_weights_only.pth")