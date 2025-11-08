<!-- markdownlint-disable MD012 MD013 MD024 MD031 MD033 MD034 MD040 MD046 -->
<!-- MD012: Multiple consecutive blank lines -->
<!-- MD013: Line length -->
<!-- MD024: Multiple headings with the same content -->
<!-- MD031: Fenced code blocks should be surrounded by blank lines -->
<!-- MD033: Inline HTML -->
<!-- MD034: Bare URL used -->
<!-- MD040: Fenced code blocks should have a language specified -->
<!-- MD046: Code block style -->

# MODELS_IMPLEMENTATION - Coding Standards

## Overview

This document defines the coding standards and best practices for implementing
model classes in the PhysicsNeMo repository. These rules are designed to ensure
consistency, maintainability, and high code quality across all model
implementations.

**Important:** These rules are enforced as strictly as possible. Deviations
from these standards should only be made when absolutely necessary and must be
documented with clear justification in code comments and approved during code
review.

## Document Organization

This document is structured in two main sections:

1. **Rule Index**: A quick-reference table listing all rules with their IDs,
   one-line summaries, and the context in which they apply. Use this section
   to quickly identify relevant rules when implementing or reviewing code.

2. **Detailed Rules**: Comprehensive descriptions of each rule, including:
   - Clear descriptions of what the rule requires
   - Rationale explaining why the rule exists
   - Examples demonstrating correct implementation
   - Anti-patterns showing common mistakes to avoid

## How to Use This Document

- **When creating new models**: Review all rules before starting implementation,
  paying special attention to rules MOD-000 through MOD-003.
- **When reviewing code**: Use the Rule Index to quickly verify compliance with
  all applicable rules.
- **When refactoring**: Ensure refactored code maintains or improves compliance
  with these standards.
- **For AI agents**: This document is formatted for easy parsing. Each rule has
  a unique ID and structured sections (Description, Rationale, Example,
  Anti-pattern) that can be extracted and used as context.

## Rule Index

| Rule ID | Summary | Apply When |
|---------|---------|------------|
| `MOD-000` | Models organization and where to place new model classes | Creating or refactoring new model classes |
| `MOD-001` | Use proper class inheritance for all models | Creating or refactoring new model classes |
| `MOD-002` | Model classes lifecycle | Creating or moving existing model classes |
| `MOD-003` | Model classes documentation | Creating or editing any docstring in a model class |

---

## Detailed Rules

### MOD-000: Models organization and where to place new model classes

**Description:**

There are two types of models in PhysicsNeMo:

- Reusable layers that are the building blocks of more complex artchitectures.
  Those should go into `physicsnemo/nn`. Those include for instance
  `FullyConnected`, various variants of attention layers, `UNetBlock` (a block
  of a U-Net), etc.
  All layers that are directly exposed to the user should be imported in
  `physicsnemo/nn/__init__.py`, such that they can be used as follows:
  ```python
  from physicsnemo.nn import MyLayer
  ```
- More complete models, composed of multiple layers and/or other sub-models.
  Those should go into `physicsnemo/models`. All models that are directly
  exposed to the user should be imported in `physicsnemo/models/__init__.py`,
  such that they can be used as follows:
  ```python
  from physicsnemo.models import MyModel
  ```

The only exception to this rule is for models or layers that are highly specific to a
single example. In this case, it may be acceptable to place them in a module
specific to the example code, such as for example
`examples/<example_name>/utils/nn.py`.

**Rationale:**
Ensures consistency and clarity in the organization of models in the
repository, in particular a clear separation between reusable layers and more
complete models that are applicable to a specific domain or specific data
modality.

---

### MOD-001: Use proper class inheritance for all models

**Description:**
All model classes must inherit from `physicsnemo.Module`. Subclasses of
`torch.nn.Module` are not allowed.
Ensure proper initialization of parent classes using `super().__init__()`. Pass
the `meta` argument to the `super().__init__()` call if appropriate, otherwise
set it manually with `self.meta = meta`.

**Rationale:**
Ensures invariants and functionality of the `physicsnemo.Module` class for all
models.

**Example:**

```python
from physicsnemo.models import Module

class MyModel(Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(meta=MyModelMetaData())
        self.linear = nn.Linear(input_dim, output_dim)
```

**Anti-pattern:**

```python
from torch import nn

class MyModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        self.linear = nn.Linear(input_dim, output_dim)
```

---

### MOD-002: Model classes lifecycle

**Description:**
All model classes must follow the following lifecycle:

- Stage 1: Creation. This is the stage where the model class is created. For
  the vast majority of models, new classes are created either in
  `physicsnemo/experimental/nn` for reusable layers, or in
  `physicsnemo/experimental/models` for more complete models. The `experimental`
  folder is used to store models that are still under development (beta or
  alpha releases) during this stage, backward compatibility is not guaranteed.
  One exception is when the developper is highly confident that the model
  is sufficiently mature andapplicable to many domains or use cases. In this
  case the model class can be created in the `physicsnemo/nn` or `physicsnemo/models`
  folders directly, and backward compatibility is guaranteed. Another exception
  is when the model class is highly specific to a single example. In this case,
  it may be acceptable to place it in a module specific to the example code,
  such as for example `examples/<example_name>/utils/nn.py`.

- Stage 2: Production. After staying in stage 1 for a sufficient amount of time
  (typically at least 1 release cycle), the model class is promoted to stage 2.
  It is then moved to the `physicsnemo/nn` or `physicsnemo/models` folders,
  based on the rule `MOD-000`. During this stage, backward compatibility is
  guaranteed.

- Stage 3: Pre-deprecation. For a model class in stage 3 in `physicsnemo/nn` or
  `physicsnemo/models`, the developper should start planning its deprecation.
  This is done by adding a warning message to the model class, indicating that
  the model class is deprecated and will be removed in a future release. The
  warning message should be a clear and concise message that explains why the
  model class is being deprecated and what the user should do instead. The
  deprecation message should be added to both the docstring and should be
  raised at runtime. The developper is free to choose the mechanism to raise the
  deprecation warning. A mnodel class cannot be deprecated without staying in
  stage 3 "pre-deprecation" for at least 1 release cycle.

- Stage 4: Deprecation. After staying in stage 3 "pre-deprecation" for at least 1
  release cycle, the model class is deprecated. It can be deleted from the
  codebase.

**Rationale:**
This lifecycle ensures a structured approach to model development and maintenance.
The experimental stage allows rapid iteration without backward compatibility
constraints, enabling developers to refine APIs based on user feedback. The
production stage provides stability for users who depend on these models. The
pre-deprecation and deprecation stages ensure users have sufficient time to
migrate to newer alternatives, preventing breaking changes that could disrupt
their workflows. This graduated approach balances innovation with stability,
a critical requirement for a scientific computing framework.

---

### MOD-003: Use proper class inheritance for all models

**Description:**

Every new model or modification of any model code should be documented with a
comprehensive docstring. Each method of the model class should be documented with a
docstring as well. The foward method should be documented in the docstring of the
model class, instead of being in the docstring of the forward method itself. A
doctrsing for the forward is still possible but it should be concise and to the
point. To document the forward method, use the sections `Forward` and
`Outputs`. In addition, all docstrings should be written in the NumPy style,
and adopt formatting to be compatible with our Sphinx restructured text (RST)
documentation.
The docstrings should follow the following requirements:

- Each docstring should be prefixed with `r"""`.

- The class docstring should at least contain three sections: `Parameters`,
  `Forward`, and `Outputs`. Other sections such as `Notes`, `Examples`,
  or `..important::` or `..code-block:: python` are possible. Other sections
  are not recognized by our Sphinx documentation and are prohibited.

- All methods should be documented with a docstring, with at least a `Parameters`
  section and a `Returns` section. Other sections such as `Notes`, `Examples`,
  or `..important::` or `..code-block:: python` are possible. Other sections
  are not recognized by our Sphinx documentation and are prohibited.

- All tensors should be documented with their shape, using LaTeX math notation such
  as :math:`(N, C, H_{in}, W_{in})` (there is flexibility for naming the
  dimensions, but te math format should be enforced). Our documentation is
  rendered using LaTeX, and supports a rich set of LaTeX commands, so
  it is recommended to use LaTeX commands whenever possible for mathematical
  variables in the docstrings. The mathematical notations should be to some degree
  consistent with the actual variable names in the code (even though
  that is not always possible, to avoid too complex formatting).

- For arguments or variables that are callback functions, (e.g. Callable), the
  docstring should include a clear sepaarted ..code-block:: that specifies the
  required signature and return type of the callback function. This is not only
  true for callback functions, but for any type of parameters or arguments that
  has some complex type specification or API requirements. The explanation code
  block should be placed in the top or bottom section of the docstrings, but
  not in the `Parameters` or `Forward` or `Outputs` sections, for readability
  and clarity.

- Inline code should be formatted double backticks, such as ```my_variable```.
  Single backticks are not allowed as they don't render properly in our Sphinx
  documentation.

- All parameters should be documented with their type and default values on a
  single line.

- When possible docstring should use links to other docstrings, such as
  :class:`~physicsnemo.models.some_model.SomeModel`, or
  :func:`~physicsnemo.utils.common_function`, or
  :meth:`~physicsnemo.models.some_model.SomeModel.some_method`.

- When referencing external resources, such as papers, websites, or other
  documentation, docstrings should use links to the external resource in the
  format `some link text <some_url>`_.

**Rationale:**
Comprehensive and well-formatted documentation is essential for scientific
software. It enables users to understand model capabilities, expected inputs,
and outputs without inspecting source code. LaTeX math notation and proper
formatting ensure documentation renders correctly in Sphinx, creating
professional, publication-quality documentation. Consistent documentation
standards facilitate automatic documentation generation, improve code
discoverability, and help AI agents understand code context. For a framework
used in scientific research, clear documentation of tensor shapes, mathematical
formulations, and API contracts is critical for reproducibility and correct usage.

**Example:**

```python
from typing import Callable, Optional
from physicsnemo.models import Module
import torch

class SimpleEncoder(Module):
    r"""
    # Rule: Docstring starts with r (for raw string) followed by three double quotes for proper LaTeX rendering
    A simple encoder network that transforms input features to a latent representation.

    This model applies a sequence of linear transformations with activation functions
    to encode input data into a lower-dimensional latent space. The architecture is
    based on the approach described in `Autoencoder Networks <https://arxiv.org/example>`_.
    # Rule: External references use proper link format

    The model supports custom preprocessing via a callback function that is applied
    to the input before encoding.

    .. code-block:: python

        # Rule: Callback functions or complex API requirements documented in a concise code-block
        def preprocess_fn(x: torch.Tensor) -> torch.Tensor:
            ...
            return y
    
    where ``x`` is the input tensor of shape :math:`(B, D_{in})` and ``y`` is the output tensor of shape :math:`(B, D_{out})`.

    Parameters
    ----------
    # Rule: Parameters section is mandatory
    input_dim : int
        # Rule: Type and description on single line
        Dimension of input features.
    latent_dim : int
        Dimension of the latent representation.
    hidden_dim : int, optional, default=128
        # Rule: Default values are documented on same line
        Dimension of the hidden layer.
    activation : str, optional, default="relu"
        Activation function to use. See :func:`~torch.nn.functional.relu` for details.
        # Rule: Cross-references use proper Sphinx syntax
    preprocess_fn : Callable[[torch.Tensor], torch.Tensor], optional, default=None
        Optional preprocessing function applied to input. See the code block above
        for the required signature.
        # Rule: Callback functions documented with reference to code-block

    Forward
    # Rule: Forward section documents the forward method in class docstring
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, D_{in})` where :math:`B` is batch size
        # Rule: Tensor shapes use LaTeX math notation with :math:
        and :math:`D_{in}` is input dimension.
    return_hidden : bool, optional, default=False
        If ``True``, also returns hidden layer activations.
        # Rule: Inline code uses double backticks

    Outputs
    # Rule: Outputs section is mandatory
    -------
    torch.Tensor or tuple
        If ``return_hidden`` is ``False``, returns latent representation of shape
        :math:`(B, D_{latent})`. If ``True``, returns tuple of
        (latent, hidden) where hidden has shape :math:`(B, D_{hidden})`.

    Examples
    # Rule: Examples section is allowed and helpful
    --------
    >>> model = SimpleEncoder(input_dim=784, latent_dim=64)
    >>> x = torch.randn(32, 784)
    >>> latent = model(x)
    >>> latent.shape
    torch.Size([32, 64])

    Notes
    # Rule: Notes section is allowed for additional context
    -----
        This encoder can be used as part of a larger autoencoder architecture
        by combining it with a decoder network such as
        :class:`~physicsnemo.models.decoder.SimpleDecoder`.
        # Rule: Cross-references to other classes use proper Sphinx syntax
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        activation: str = "relu",
        preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        super().__init__(meta=SimpleEncoderMetaData())
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, latent_dim)
        self.preprocess_fn = preprocess_fn

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        """Concise forward docstring referencing class docstring for details."""
        # Rule: Forward method can have concise docstring, main docs in class
        if self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
        hidden = torch.relu(self.fc1(x))
        latent = self.fc2(hidden)
        if return_hidden:
            return latent, hidden
        return latent

    def compute_reconstruction_loss(
        self,
        latent: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute mean squared error between latent representation and target.
        # Rule: Methods have their own docstrings with Parameters and Returns sections

        Parameters
        ----------
        # Rule: Parameters section is mandatory for methods
        latent : torch.Tensor
            Latent representation of shape :math:`(B, D_{latent})`.
            # Rule: Tensor shapes documented with :math:
        target : torch.Tensor
            Target tensor of shape :math:`(B, D_{latent})`.

        Returns
        -------
        # Rule: Returns section is mandatory for methods (not "Outputs" - that's for forward)
        torch.Tensor
            Scalar loss value.
        """
        return torch.nn.functional.mse_loss(latent, target)
```

**Anti-pattern:**

```python
from physicsnemo.models import Module
import torch

class BadEncoder(Module):
    '''
    # WRONG: Should use r (for raw string) followed by three double quotes for docstrings not three single quotes
    A simple encoder network
    # WRONG: missing Parameters, Forward, or Outputs sections in the docstring
    # WRONG: callback function preprocess_fn is not documented at all
    '''
    def __init__(self, input_dim, latent_dim, hidden_dim=128, preprocess_fn=None):
        # WRONG: No type hints in signature
        # WRONG: preprocess_fn callback parameter is missing from docstring entirely
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, latent_dim)
        self.preprocess_fn = preprocess_fn

    def forward(self, x, return_hidden=False):
        # WRONG: No type hints
        """
        Forward pass of the encoder.
        # WRONG: Should be brief, main docs for the forward method should be in the class docstring

        Args:
            x: input tensor with shape (B, D_in)
            # WRONG: Using "Args" instead of NumPy-style "Parameters"
            # WRONG: Not using :math: with backticks for shapes and other mathematical notations
            return_hidden (bool): whether to return hidden state
            # WRONG: Mixing documentation styles

        Returns:
            encoded representation
            # WRONG: Using "Returns" instead of "Outputs". The "Returns" section is for methods other than the forward method. The forward method should have an "Outputs" section instead.
            # WRONG: No shape information
        """
        # WRONG: Entire forward signature and behavior should be in class docstring
        if self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
        hidden = torch.relu(self.fc1(x))
        latent = self.fc2(hidden)
        if return_hidden:
            return latent, hidden
        return latent

    def compute_loss(self, x, y):
        # WRONG: No type hints
        """
        Compute the loss
        # WRONG: No proper docstring structure

        Description:
            # WRONG: "Description" is not a recognized section, should be in the main description
            This method computes some loss value

        Arguments:
            # WRONG: "Arguments" is not recognized, should be "Parameters"
            x: first input (B, D)
            # WRONG: Not using :math: for tensor shapes
            y: second input
            # WRONG: No shape information

        Output:
            # WRONG: "Output" (singular) is not recognized, should be "Returns" for methods
            loss value
            # WRONG: No type or shape information
        """
        return torch.nn.functional.mse_loss(x, y)

    def helper_method(self, x):
        # WRONG: No docstring at all for method, docstrings for methods are mandatory
        # WRONG: missing Parameters and Returns sections in the docstring
        # WRONG: No type hints
        return x * 2
```

---



## Compliance

When implementing models, ensure all rules are followed. Code reviews should
verify each rule is followed and enforce the rules as strictly as possible.
For exceptions to these rules, document the reasoning in code comments and
obtain approval during code review.
