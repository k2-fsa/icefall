#!/usr/bin/env python3
#
# Copyright      2022-2023  Xiaomi Corp.        (authors: Fangjun Kuang)
import ncnn
import numpy as np


layer_list = []


def RegisterCustomLayers(net):
    RegisterPoolingModuleNoProj(net)
    RegisterTensorAsStrided(net)
    RegisterSimpleUpsample(net)
    RegisterStack(net)


def RegisterPoolingModuleNoProj(net):
    net.register_custom_layer(
        "PoolingModuleNoProj",
        PoolingModuleNoProjCreator,
        PoolingModuleNoProjDeleter,
    )


def PoolingModuleNoProjCreator():
    return PoolingModuleNoProj()


def PoolingModuleNoProjDeleter(l):
    for i, layer in enumerate(layer_list):
        if layer == l:
            del layer_list[i]
            break


def TensorAsStridedCreator():
    return TensorAsStrided()


def TensorAsStridedDeleter(l):
    for i, layer in enumerate(layer_list):
        if layer == l:
            del layer_list[i]
            break


def RegisterTensorAsStrided(net):
    net.register_custom_layer(
        "TensorAsStrided",
        TensorAsStridedCreator,
        TensorAsStridedDeleter,
    )


def SimpleUpsampleCreator():
    return SimpleUpsample()


def SimpleUpsampleDeleter(l):
    for i, layer in enumerate(layer_list):
        if layer == l:
            del layer_list[i]
            break


def RegisterSimpleUpsample(net):
    net.register_custom_layer(
        "SimpleUpsample",
        SimpleUpsampleCreator,
        SimpleUpsampleDeleter,
    )


def StackCreator():
    return Stack()


def StackDeleter(l):
    for i, layer in enumerate(layer_list):
        if layer == l:
            del layer_list[i]
            break


def RegisterStack(net):
    net.register_custom_layer(
        "Stack",
        StackCreator,
        StackDeleter,
    )


class PoolingModuleNoProj(ncnn.Layer):
    def __init__(self):
        super().__init__()
        self.one_blob_only = False
        self.support_inplace = False
        layer_list.append(self)

    def forward(self, bottom_blobs, top_blobs, opt):
        x = bottom_blobs[0]
        cached_len = bottom_blobs[1]
        cached_avg = bottom_blobs[2]

        # x.dims = 2, x.w = C, x.h = T, e.g., C=384, T=16
        # cached_len.dims = 1, cached_len.w = 1
        # cached_avg.dims = 2, cached_avg.w = C, cached_len.h = 1, e.g., C=384

        x = x.numpy()  # x is of shape (T, C), e.g., (16, 384)
        x = x.cumsum(axis=0)

        cached_len = cached_len.numpy()
        cached_avg = cached_avg.numpy()

        x = x + cached_len * cached_avg[0]
        scale = np.arange(1, x.shape[0] + 1, dtype=np.float32).reshape(-1, 1)
        x = x / (scale + cached_len)

        out_cached_len = cached_len + x.shape[0]
        out_cached_avg = x[-1:]

        top_blobs[0].clone_from(ncnn.Mat(x), opt.blob_allocator)
        top_blobs[1].clone_from(ncnn.Mat(out_cached_len), opt.blob_allocator)
        top_blobs[2].clone_from(ncnn.Mat(out_cached_avg), opt.blob_allocator)

        #  print(top_blobs[0].numpy().shape)
        #  print(top_blobs[1].numpy().shape)
        #  print(top_blobs[2].numpy().shape)
        return 0


class TensorAsStrided(ncnn.Layer):
    def __init__(self):
        super().__init__()
        self.one_blob_only = True
        self.support_inplace = False

        layer_list.append(self)

    def load_param(self, pd):
        sizes = pd.get(0, ncnn.Mat())
        strides = pd.get(1, ncnn.Mat())
        storage_offset = pd.get(2, 0)

        assert sizes.dims == 1, sizes.dims
        assert strides.dims == 1, strides.dims

        assert sizes.w == strides.w, (sizes.w, strides.w)

        self.sizes = sizes.numpy("i").tolist()
        self.strides = strides.numpy("i").tolist()
        self.storage_offset = storage_offset

        return 0

    def forward(self, bottom_blob, top_blob, opt):
        if bottom_blob.dims != 3:
            raise ValueError(
                f"Only 3-D tensors are supported. Given {bottom_blob.dims}"
            )
        in_c = bottom_blob.c
        in_h = bottom_blob.h
        in_w = bottom_blob.w

        out_c = self.sizes[0]
        out_h = self.sizes[1]
        out_w = self.sizes[2]

        assert in_c == out_c, (in_c, out_c)
        assert self.strides[0] == in_h * in_w, (
            self.strides[0],
            in_h,
            in_w,
            in_h * in_w,
        )

        bottom_blob = bottom_blob.numpy()
        out = np.empty((out_c, out_h, out_w), dtype=np.float32)

        for c in range(out_c):
            p = bottom_blob[c].reshape(-1)[self.storage_offset :]
            for h in range(out_h):
                q = p[h * self.strides[1] :]
                if True:
                    for w in range(out_w):
                        out[c][h][w] = q[w * self.strides[2]]
                else:
                    out[c][h] = q[: (out_w * self.strides[2]) : self.strides[2]]

        top_blob.clone_from(ncnn.Mat(out), opt.blob_allocator)

        return 0


class SimpleUpsample(ncnn.Layer):
    def __init__(self):
        super().__init__()
        self.one_blob_only = True
        self.support_inplace = False

        layer_list.append(self)

    def load_param(self, pd):
        upsample = pd.get(0, 0)
        num_channels = pd.get(1, 0)
        bias_data_size = pd.get(2, 0)

        assert upsample * num_channels == bias_data_size, (
            upsample,
            num_channels,
            bias_data_size,
            upsample * num_channels,
        )

        self.upsample = upsample
        self.num_channels = num_channels
        self.bias_data_size = bias_data_size

        return 0

    def load_model(self, md):
        bias = md.load(self.num_channels, self.upsample, 0)
        assert bias.w == self.num_channels, (bias.w, self.num_channels)
        assert bias.h == self.upsample, (bias.h, self.upsample)

        self.bias = bias.numpy()  # its shape is (upsample, num_channels)

        return 0

    def forward(self, bottom_blob, top_blob, opt):
        assert bottom_blob.dims == 2, bottom_blob.dims
        assert bottom_blob.w == self.num_channels, (bottom_blob.w, self.num_channels)

        bottom_blob = bottom_blob.numpy()

        out = np.expand_dims(bottom_blob, axis=1) + self.bias
        out = out.reshape(-1, self.num_channels)

        top_blob.clone_from(ncnn.Mat(out), opt.blob_allocator)

        return 0


class Stack(ncnn.Layer):
    def __init__(self):
        super().__init__()
        self.one_blob_only = False
        self.support_inplace = False

        layer_list.append(self)

    def load_param(self, pd):
        axis = pd.get(0, 0)

        self.axis = axis

        return 0

    def forward(self, bottom_blobs, top_blobs, opt):
        bottom_blobs = [b.numpy() for b in bottom_blobs]
        out = np.stack(bottom_blobs, axis=self.axis)

        top_blobs[0].clone_from(ncnn.Mat(out), opt.blob_allocator)

        return 0
