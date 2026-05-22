# Licensed under the MIT License. See LICENSE file for details.
# Copyright (c) 2026 Zhenyu Zhao

import torch
from typing import Literal, Optional

from torch.nn.parameter import Parameter
import math
from .utils import unitary_from_H, batch_unitary_cayley
from torch.nn.common_types import _size_2_t
import torch.nn.functional as F


class PhotonicLayer(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device=None,
            dtype=torch.complex64
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n = self.in_features if self.in_features >= self.out_features else self.out_features
        self.weight = Parameter(
            torch.ones((self.n, self.n), **factory_kwargs) * math.pi / 4
        )

    def forward(self, input):
        U = unitary_from_H(self.weight)

        if self.in_features >= self.out_features:
            input_v = torch.exp(1j * input * math.pi)
        else:
            ones = torch.ones(input.shape[:-1] + (self.n - self.in_features,), device=input.device, dtype=input.dtype)
            input_v = torch.cat([torch.exp(1j * input * math.pi), ones], dim=-1)

        output = torch.matmul(input_v, U.T)
        output_intensity = torch.pow(torch.abs(output), 2) / self.in_features
        if self.in_features >= self.out_features:
            return output_intensity[..., :self.out_features]
        else:
            return output_intensity


class NLPhotonicLayer(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device=None,
            dtype=torch.complex64
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n = self.in_features + 1 if self.in_features >= self.out_features else (self.out_features + 1)
        self.weight = Parameter(
            torch.ones((self.n, self.n), **factory_kwargs) * math.pi / 4
        )

    def forward(self, input):
        U = unitary_from_H(self.weight)
        if self.in_features >= self.out_features:
            ones = torch.ones(input.shape[:-1] + (1,), device=input.device, dtype=input.dtype)
            input_v = torch.cat([torch.exp(1j * input * math.pi), ones], dim=-1)
        else:
            ones = torch.ones(input.shape[:-1] + (1,), device=input.device, dtype=input.dtype)
            input_v = torch.cat([torch.exp(1j * input * math.pi), ones], dim=-1)
            zeros = torch.ones(input.shape[:-1] + (self.n - self.in_features - 1,), device=input.device, dtype=input.dtype)
            input_v = torch.cat([input_v, zeros], dim=-1)

        output = torch.matmul(input_v, U.T)
        output_intensity = torch.pow(torch.abs(output), 2) / self.n

        return output_intensity[..., :self.out_features]


class PhotonicConv2DLayer(torch.nn.Module):
    """
    reference: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: str or _size_2_t = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if isinstance(padding, str):
            raise ValueError(
                f"Invalid padding string."
            )

        valid_padding_modes = {"zeros"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_group_size = self.in_channels // self.groups
        self.out_group_size = self.out_channels // self.groups
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.padding_mode = padding_mode

        self.kH, self.kW = kernel_size

        self.weight = Parameter(
            torch.ones((self.groups, self.out_group_size, self.in_group_size*self.kH * self.kW,
                        self.in_group_size*self.kH * self.kW), **factory_kwargs) * math.pi / 4
        )

    def forward(self, input):
        U = unitary_from_H(self.weight)
        complex_weight = U[:, :, 0, :]
        N, C_in, H, W = input.shape
        # im2col
        x_unfold = F.unfold(input, (self.kH, self.kW), self.dilation, self.padding, self.stride)
        L = x_unfold.shape[2]
        x_unfold_grouped = x_unfold.view(
            N, self.groups, self.in_group_size * self.kH * self.kW, L
        )
        cols_v = torch.exp(1j * x_unfold_grouped * math.pi)
        output_grouped = complex_weight @ cols_v
        # reshape
        H_out = (H + 2 * self.padding[0] - self.dilation[0] * (self.kH - 1) - 1) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.dilation[1] * (self.kW - 1) - 1) // self.stride[1] + 1
        output = output_grouped.transpose(1, 2).contiguous()
        output = output.view(N, self.out_channels, L)
        out = output.view(N, self.out_channels, H_out, W_out)
        output_intensity = torch.pow(torch.abs(out), 2) / (self.in_group_size*self.kH * self.kW)

        return output_intensity


class NLPhotonicConv2DLayer(torch.nn.Module):
    """
    reference: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: str or _size_2_t = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
            device=None,
            dtype=torch.complex64,
    ) -> None:
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if isinstance(padding, str):
            raise ValueError(
                f"Invalid padding string."
            )

        valid_padding_modes = {"zeros"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_group_size = self.in_channels // self.groups
        self.out_group_size = self.out_channels // self.groups
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.padding_mode = padding_mode

        self.kH, self.kW = kernel_size

        self.weight = Parameter(
            torch.ones((self.groups, self.out_group_size, self.in_group_size*self.kH * self.kW + 1,
                        self.in_group_size*self.kH * self.kW + 1), **self.factory_kwargs) * math.pi / 4
        )

        self.ones = None
        self.ones_shape = None

    def forward(self, input):
        U = unitary_from_H(self.weight)
        complex_weight = U[:, :, 0, :]

        N, C_in, H, W = input.shape
        # im2col
        x_unfold = F.unfold(input, (self.kH, self.kW), self.dilation, self.padding, self.stride)
        L = x_unfold.shape[2]
        x_unfold_grouped = x_unfold.view(
            N, self.groups, self.in_group_size * self.kH * self.kW, L
        )
        ones_shape = list(x_unfold_grouped.shape)
        ones_shape[-2] = 1
        if self.ones is None:
            self.ones_shape = ones_shape
            self.ones = torch.ones(self.ones_shape, device=x_unfold_grouped.device, dtype=x_unfold_grouped.dtype)
            cols_v = torch.cat([torch.exp(1j * x_unfold_grouped * math.pi), self.ones], dim=-2)
        else:
            if tuple(ones_shape) != tuple(self.ones_shape):
                self.ones_shape = ones_shape
                self.ones = torch.ones(self.ones_shape, device=x_unfold_grouped.device, dtype=x_unfold_grouped.dtype)
                cols_v = torch.cat([torch.exp(1j * x_unfold_grouped * math.pi), self.ones], dim=-2)
            else:
                cols_v = torch.cat([torch.exp(1j * x_unfold_grouped * math.pi), self.ones], dim=-2)
        output_grouped = complex_weight @ cols_v
        # reshape
        H_out = (H + 2 * self.padding[0] - self.dilation[0] * (self.kH - 1) - 1) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.dilation[1] * (self.kW - 1) - 1) // self.stride[1] + 1
        output = output_grouped.transpose(1, 2).contiguous()
        output = output.view(N, self.out_channels, L)
        out = output.view(N, self.out_channels, H_out, W_out)
        output_intensity = torch.pow(torch.abs(out), 2) / (self.in_group_size*self.kH * self.kW + 1)

        return output_intensity


class SPhotonicConv2DLayer(torch.nn.Module):
    """
    reference: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: str or _size_2_t = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if isinstance(padding, str):
            raise ValueError(
                f"Invalid padding string."
            )

        valid_padding_modes = {"zeros"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_group_size = self.in_channels // self.groups
        self.out_group_size = self.out_channels // self.groups
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.padding_mode = padding_mode

        self.kH, self.kW = kernel_size

        D = self.in_group_size * self.kH * self.kW
        scale = 1.0 / math.sqrt(D)

        real = torch.ones(self.groups, self.out_group_size, D, **factory_kwargs)
        imag = torch.ones(self.groups, self.out_group_size, D, **factory_kwargs)

        self.weight = Parameter(
            scale * (real + 1j * imag)
        )


    def forward(self, input):
        complex_weight = self.weight / self.weight.norm(dim=-1, keepdim=True)
        N, C_in, H, W = input.shape
        # im2col
        x_unfold = F.unfold(input, (self.kH, self.kW), self.dilation, self.padding, self.stride)
        L = x_unfold.shape[2]
        x_unfold_grouped = x_unfold.view(
            N, self.groups, self.in_group_size * self.kH * self.kW, L
        )
        cols_v = torch.exp(1j * x_unfold_grouped * math.pi)
        output_grouped = complex_weight @ cols_v
        # reshape
        H_out = (H + 2 * self.padding[0] - self.dilation[0] * (self.kH - 1) - 1) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.dilation[1] * (self.kW - 1) - 1) // self.stride[1] + 1
        output = output_grouped.transpose(1, 2).contiguous()
        output = output.view(N, self.out_channels, L)
        out = output.view(N, self.out_channels, H_out, W_out)
        output_intensity = torch.pow(torch.abs(out), 2) / (self.in_group_size * self.kH * self.kW)

        return output_intensity


class SNLPhotonicConv2DLayer(torch.nn.Module):
    """
    reference: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: str or _size_2_t = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
            device=None,
            dtype=torch.complex64,
    ) -> None:
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if isinstance(padding, str):
            raise ValueError(
                f"Invalid padding string."
            )

        valid_padding_modes = {"zeros"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_group_size = self.in_channels // self.groups
        self.out_group_size = self.out_channels // self.groups
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.padding_mode = padding_mode

        self.kH, self.kW = kernel_size
        D = self.in_group_size * self.kH * self.kW
        scale = 1.0 / math.sqrt(D + 1)

        real = torch.ones(self.groups, self.out_group_size, D + 1, **self.factory_kwargs)
        imag = torch.ones(self.groups, self.out_group_size, D + 1, **self.factory_kwargs)

        self.weight = Parameter(
            scale * (real + 1j * imag)
        )

        self.ones = None
        self.ones_shape = None

    def forward(self, input):
        complex_weight = self.weight / self.weight.norm(dim=-1, keepdim=True)
        N, C_in, H, W = input.shape
        # im2col
        x_unfold = F.unfold(input, (self.kH, self.kW), self.dilation, self.padding, self.stride)
        L = x_unfold.shape[2]
        x_unfold_grouped = x_unfold.view(
            N, self.groups, self.in_group_size * self.kH * self.kW, L
        )
        ones_shape = list(x_unfold_grouped.shape)
        ones_shape[-2] = 1
        if self.ones is None:
            self.ones_shape = ones_shape
            self.ones = torch.ones(self.ones_shape, device=x_unfold_grouped.device, dtype=x_unfold_grouped.dtype)
            cols_v = torch.cat([torch.exp(1j * x_unfold_grouped * math.pi), self.ones], dim=-2)
        else:
            if tuple(ones_shape) != tuple(self.ones_shape):
                self.ones_shape = ones_shape
                self.ones = torch.ones(self.ones_shape, device=x_unfold_grouped.device, dtype=x_unfold_grouped.dtype)
                cols_v = torch.cat([torch.exp(1j * x_unfold_grouped * math.pi), self.ones], dim=-2)
            else:
                cols_v = torch.cat([torch.exp(1j * x_unfold_grouped * math.pi), self.ones], dim=-2)
        output_grouped = complex_weight @ cols_v
        # reshape
        H_out = (H + 2 * self.padding[0] - self.dilation[0] * (self.kH - 1) - 1) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.dilation[1] * (self.kW - 1) - 1) // self.stride[1] + 1
        output = output_grouped.transpose(1, 2).contiguous()
        output = output.view(N, self.out_channels, L)
        out = output.view(N, self.out_channels, H_out, W_out)
        output_intensity = torch.pow(torch.abs(out), 2) / (self.in_group_size * self.kH * self.kW)

        return output_intensity