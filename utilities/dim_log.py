def tensor_info(name, tensor):
    dims = tensor.size()
    print(f'T({name}): {" x ".join(map(str, tensor.detach().shape))}')