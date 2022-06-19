import numpy as np
import torch

import os
from pathlib import Path


def select(l, sel) :
    if sel is None :
        return []
    if type(sel) is bool :
        if sel : return l
        else : return []
    elif hasattr(sel, '__len__') :
        return [l[i] for i in sel]
    elif type(sel) is int :
        if sel > 0 :
            return l[:sel]
        elif sel < 0 :
            return l[sel:]
        else :
            return []
    else :
        raise NotImplementedError()


def collate_dict_adaptive(batch_list, pad_keys_if_necessary=None) :
    if pad_keys_if_necessary is None :
        pad_keys_if_necessary = []
    res = {}
    for k in batch_list[0].keys() :
        k_batch = [i[k] for i in batch_list]
        if type(k_batch[0]) is np.ndarray :
            dtype = torch.float32 if k_batch[0].dtype == np.float64 else None
            k_tensors = [torch.tensor(arr, dtype=dtype) for arr in k_batch]
            ls = set(map(lambda x : x.shape[0], k_tensors))
            if len(ls) == 1 :
                k_batch = torch.stack(k_tensors)
            else :
                if k in pad_keys_if_necessary:
                    k_batch = torch.zeros((len(k_tensors), max(ls), *k_tensors[0].shape[1:]),
                                          dtype=k_tensors[0].dtype)
                    k_batch_mask = torch.zeros(k_batch.shape[:2], dtype=bool)
                    for i, t in enumerate(k_tensors) :
                        k_batch[i, :t.shape[0]] = t
                        k_batch_mask[i, :t.shape[0]] = True
                    res[k + "_pad_mask"] = k_batch_mask
                else :
                    k_batch = k_tensors

        res[k] = k_batch

    return res



def collate_dict_arrays(batch_list) :
    assert type(batch_list[0]) is dict
    return collate_dict_arrays_descent(batch_list)

def collate_dict_arrays_descent(dict_list) :
    res = {}
    for k in dict_list[0].keys() :
        sub_list = [dict_list[i][k] for i in range(len(dict_list))]
        if type(dict_list[0][k]) is dict :
            res[k] = collate_dict_arrays_descent(sub_list)
        elif type(dict_list[0][k]) is np.ndarray :
            res[k] = torch.stack([torch.tensor(arr, dtype=torch.float32) if arr.dtype==np.float64 else
                                  torch.tensor(arr) for arr in sub_list])
        else :
            res[k] = sub_list
    return res



def collate_dict_list(batch_list) :
    # batch is iterable
    if type(batch_list[0]) is list or type(batch_list[0]) is tuple :
        assert False, "only want to collate list of dicts here"
        batch_list_flat = []
        for updict in batch_list :
            for subdict in updict :
                batch_list_flat.append(subdict)
        batch_list = batch_list_flat
    assert type(batch_list[0]) is dict
    return collate_dict_descent(batch_list)


def collate_dict_descent(dict_list) :
    res = {}
    for k in dict_list[0].keys():
        sub_list = [dict_list[i][k] for i in range(len(dict_list))]
        if type(dict_list[0][k]) is dict:
            res[k] = collate_dict_descent(sub_list)
        elif type(dict_list[0][k]) is np.ndarray:
            res[k] = [torch.tensor(arr, dtype=torch.float32) if arr.dtype==np.float64 else
                      torch.tensor(arr) for arr in sub_list]
        else:
            res[k] = sub_list
    return res

def tensor_or_tensorlist_to_device(batch, device, tensor_keys=None) :
    if tensor_keys is None :
        tensor_keys = batch.keys()
    for key in tensor_keys :
        if type(batch[key]) is list :
            batch[key] = [ti.to(device) for ti in batch[key]]
        else :
            batch[key] = batch[key].to(device)
    return batch

def get_folder_depth(dir, depth) :
    dir = Path(dir)
    dir_content = sorted(os.listdir(dir))
    folders = [f for f in dir_content if os.path.isdir(dir / f)]
    if depth == 0 :
        return folders
    else :
        folders_all = []
        for f in folders :
            folders_sub = get_folder_depth(dir / f, depth-1)
            folders_sub = [f + "/" + sub for sub in folders_sub]
            folders_all += folders_sub
        return folders_all