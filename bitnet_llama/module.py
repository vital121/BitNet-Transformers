# # coding=utf-8
# # Copyright 2023 Beomi (L. Junbum)
# # Licensed under the Apache License, Version 2.0 (the "License")
""" PyTorch BitLinear Layer."""
import torch
import torch.nn as nn


class BitLinearNaive(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, num_groups=1):
        super(BitLinearNaive, self).__init__(in_features, out_features, bias)
        self.num_groups = num_groups
        self.eps = 1e-5  # Small epsilon value to avoid division by zero and overflow

    def ternarize_weights(self):
        return BitLinear.quantize_tensor(self.weight, self.eps)

    def quantize_activations(self, x: torch.Tensor, b: int = 8):
        Q_b = 2 ** (b - 1)
        gamma = x.abs().max()
        quantized_x = torch.clamp(
            x * Q_b / (gamma + self.eps), -Q_b + self.eps, Q_b - self.eps
        )
        return quantized_x

    def forward(self, input):
        # ternarize weights
        ternarized_weights = self.ternarize_weights()

        # Normal linear transformation with ternarized weights
        output = torch.nn.functional.linear(input, ternarized_weights, self.bias)

        # Quantize activations (before non-linear functions like ReLU)
        output = self.quantize_activations(output)

        return output


class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, num_groups=1):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.num_groups = num_groups
        self.eps = 1e-5

    @staticmethod
    def quantize_tensor(x: torch.Tensor, eps: float):
        return torch.clamp(
            torch.round(x / (x.mean() + eps)),
            -1,
             1,
        )

    def ste_ternarize(self, x):
        # Apply the sign function for ternarization
        ternarized_x = BitLinear.quantize_tensor(x, self.eps)
        # Use STE: during backward pass, we bypass the ternarization
        ternarized_x = (ternarized_x - x).detach() + x
        return ternarized_x

    def ternarize_weights_groupwise(self):
        # Divide weights into groups
        group_size = self.weight.shape[0] // self.num_groups
        ternarized_weights = torch.zeros_like(self.weight)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = self.weight[start_idx:end_idx]

            # Ternarize weights
            ternarized_weights[start_idx:end_idx] = self.ste_ternarize(weight_group)

        return ternarized_weights

    def quantize_activations_groupwise(self, x, b=8):
        Q_b = 2 ** (b - 1)

        # Divide activations into groups
        group_size = x.shape[0] // self.num_groups
        quantized_x = torch.zeros_like(x)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[start_idx:end_idx]

            # Quantize each group
            gamma_g = activation_group.abs().max()
            quantized_x[start_idx:end_idx] = torch.clamp(
                activation_group * Q_b / (gamma_g + self.eps),
                -Q_b + self.eps,
                Q_b - self.eps,
            )

        return quantized_x

    def forward(self, input):
        # ternarize weights (group-wise) using STE
        ternarized_weights = self.ternarize_weights_groupwise()

        # Normal linear transformation with ternarized weights
        output = torch.nn.functional.linear(input, ternarized_weights, self.bias)

        # Quantize activations group-wise
        output = self.quantize_activations_groupwise(output)

        return output


class BitLinearOptimized(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, num_groups=1):
        super(BitLinearOptimized, self).__init__(in_features, out_features, bias)
        self.num_groups = num_groups
        self.eps = 1e-5

        # Initialize 1-bit quantized weights and store them as int8
        self.register_buffer(
            "quantized_weights", BitLinear.quantize_tensor(self.weight.data).to(torch.int8)
        )
        # Clear the original weights to save memory
        del self.weight

    @property
    def weight(self):
        # Return the dequantized weights when accessed
        return self.dequantize_weights()

    @weight.setter
    def weight(self, value):
        # Update the quantized_weights when the weight property is set
        self.quantized_weights.data = BitLinear.quantize_tensor(value, self.eps)

    def dequantize_weights(self):
        # Convert quantized_weights back to bfloat16 and compute alpha for the weights
        bfloat16_weights = self.quantized_weights.to(torch.bfloat16)
        alpha = bfloat16_weights.mean()
        return bfloat16_weights * alpha

    def ste_ternarize(self, x):
        # Apply the sign function for ternarization
        ternarized_x = BitLinear.quantize_tensor(x, self.eps)
        # Use STE: during backward pass, we bypass the ternarization
        ternarized_x = (ternarized_x - x).detach() + x
        return ternarized_x

    def ternarize_weights_groupwise(self):
        # Dequantize the weights before ternarization
        weights = self.dequantize_weights()

        # Divide weights into groups
        group_size = weights.shape[0] // self.num_groups
        ternarized_weights = torch.zeros_like(weights)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = weights[start_idx:end_idx]

            # ternarize each group using STE
            ternarized_weights[start_idx:end_idx] = self.ste_ternarize(weight_group)

        return ternarized_weights

    def quantize_activations_groupwise(self, x, b=8):
        Q_b = 2 ** (b - 1)

        # Divide activations into groups
        group_size = x.shape[0] // self.num_groups
        quantized_x = torch.zeros_like(x)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[start_idx:end_idx]

            # Quantize each group
            gamma_g = activation_group.abs().max()
            quantized_x[start_idx:end_idx] = torch.clamp(
                activation_group * Q_b / (gamma_g + self.eps),
                -Q_b + self.eps,
                Q_b - self.eps,
            )

        return quantized_x

    def forward(self, input):
        # ternarize weights (group-wise) using STE
        ternarized_weights = self.ternarize_weights_groupwise()

        # Normal linear transformation with ternarized weights
        output = torch.nn.functional.linear(input, ternarized_weights, self.bias)

        # Quantize activations group-wise
        output = self.quantize_activations_groupwise(output)

        return output
